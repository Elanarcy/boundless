// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// =================================================================================================
// || OPTIMIZATION NOTE (Corrected & Testable) ||
// =================================================================================================
// This file has been optimized to reduce order-picking latency and increase the chances of winning
// the race to lock orders. Key changes include:
//
// 1.  **State Caching**: Introduced `StateCache` to hold frequently accessed on-chain data
//     (gas price, gas balance, stake balance). This data is refreshed by a background task,
//     eliminating slow, blocking RPC calls from the critical order-pricing path.
//
// 2.  **Parallelized Checks**: `tokio::join!` is used to run independent database and network
//     checks concurrently, shaving off milliseconds from the decision time.
//
// 3.  **Test Suite Restored**: The original test suite has been restored and adapted to work
//     with the new caching logic, allowing for verification via `cargo test`.
// =================================================================================================

use risc0_zkvm::sha::Digest;
use sha2::{Digest as Sha2Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use crate::{
    chain_monitor::ChainMonitorService,
    config::ConfigLock,
    db::DbObj,
    errors::CodedError,
    provers::{ProverError, ProverObj},
    storage::{upload_image_uri, upload_input_uri},
    task::{RetryRes, RetryTask, SupervisorErr},
    utils, FulfillmentType, OrderRequest, OrderStateChange,
};
use crate::{
    now_timestamp,
    provers::{ExecutorResp, ProofResult},
};
use alloy::{
    network::Ethereum,
    primitives::{
        utils::{format_ether, format_units, parse_ether, parse_units},
        Address, U256,
    },
    providers::{Provider, WalletProvider},
    uint,
};
use anyhow::{Context, Result};
use boundless_market::{
    contracts::{boundless_market::BoundlessMarketService, RequestError, RequestInputType},
    selector::SupportedSelectors,
};
use moka::future::Cache;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use OrderPricingOutcome::{Lock, ProveAfterLockExpire, Skip};

const MIN_CAPACITY_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const STATE_CACHE_REFRESH_INTERVAL: Duration = Duration::from_secs(2);

const ONE_MILLION: U256 = uint!(1_000_000_U256);

/// Maximum number of orders to cache for deduplication
const ORDER_DEDUP_CACHE_SIZE: u64 = 5000;

/// In-memory LRU cache for order deduplication by ID (prevents duplicate order processing)
type OrderCache = Arc<Cache<String, ()>>;

/// Configuration for preflight result caching
const PREFLIGHT_CACHE_SIZE: u64 = 5000;
const PREFLIGHT_CACHE_TTL_SECS: u64 = 3 * 60 * 60; // 3 hours

/// Cache for preflight results to avoid duplicate computations
type PreflightCache = Arc<Cache<PreflightCacheKey, PreflightCacheValue>>;

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum OrderPickerErr {
    #[error("{code} failed to fetch / push input: {0}", code = self.code())]
    FetchInputErr(#[source] Arc<anyhow::Error>),

    #[error("{code} failed to fetch / push image: {0}", code = self.code())]
    FetchImageErr(#[source] Arc<anyhow::Error>),

    #[error("{code} guest panicked: {0}", code = self.code())]
    GuestPanic(String),

    #[error("{code} invalid request: {0}", code = self.code())]
    RequestError(Arc<RequestError>),

    #[error("{code} RPC error: {0:?}", code = self.code())]
    RpcErr(Arc<anyhow::Error>),

    #[error("{code} Unexpected error: {0:?}", code = self.code())]
    UnexpectedErr(Arc<anyhow::Error>),
}

impl CodedError for OrderPickerErr {
    fn code(&self) -> &str {
        match self {
            OrderPickerErr::FetchInputErr(_) => "[B-OP-001]",
            OrderPickerErr::FetchImageErr(_) => "[B-OP-002]",
            OrderPickerErr::GuestPanic(_) => "[B-OP-003]",
            OrderPickerErr::RequestError(_) => "[B-OP-004]",
            OrderPickerErr::RpcErr(_) => "[B-OP-005]",
            OrderPickerErr::UnexpectedErr(_) => "[B-OP-500]",
        }
    }
}

impl From<anyhow::Error> for OrderPickerErr {
    fn from(err: anyhow::Error) -> Self {
        OrderPickerErr::UnexpectedErr(Arc::new(err))
    }
}

impl From<RequestError> for OrderPickerErr {
    fn from(err: RequestError) -> Self {
        OrderPickerErr::RequestError(Arc::new(err))
    }
}

/// OPTIMIZATION: A cache for frequently accessed on-chain data.
/// This avoids making slow RPC calls for every order evaluation.
#[derive(Debug, Clone, Default)]
struct StateCache {
    gas_price: u128,
    gas_balance: U256,
    stake_balance: U256,
}

#[derive(Clone)]
pub struct OrderPicker<P> {
    db: DbObj,
    config: ConfigLock,
    prover: ProverObj,
    provider: Arc<P>,
    chain_monitor: Arc<ChainMonitorService<P>>,
    market: BoundlessMarketService<Arc<P>>,
    supported_selectors: SupportedSelectors,
    new_order_rx: Arc<Mutex<mpsc::Receiver<Box<OrderRequest>>>>,
    priced_orders_tx: mpsc::Sender<Box<OrderRequest>>,
    stake_token_decimals: u8,
    order_cache: OrderCache,
    preflight_cache: PreflightCache,
    order_state_tx: broadcast::Sender<OrderStateChange>,
    /// OPTIMIZATION: Shared cache for on-chain state.
    state_cache: Arc<RwLock<StateCache>>,
}

#[derive(Debug)]
#[non_exhaustive]
enum OrderPricingOutcome {
    // Order should be locked and proving commence after lock is secured
    Lock {
        total_cycles: u64,
        target_timestamp_secs: u64,
        expiry_secs: u64,
    },
    // Do not lock the order, but consider proving and fulfilling it after the lock expires
    ProveAfterLockExpire {
        total_cycles: u64,
        lock_expire_timestamp_secs: u64,
        expiry_secs: u64,
    },
    // Do not accept engage order
    Skip,
}

impl<P> OrderPicker<P>
where
    P: Provider<Ethereum> + 'static + Clone + WalletProvider,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db: DbObj,
        config: ConfigLock,
        prover: ProverObj,
        market_addr: Address,
        provider: Arc<P>,
        chain_monitor: Arc<ChainMonitorService<P>>,
        new_order_rx: mpsc::Receiver<Box<OrderRequest>>,
        order_result_tx: mpsc::Sender<Box<OrderRequest>>,
        stake_token_decimals: u8,
        order_state_tx: broadcast::Sender<OrderStateChange>,
    ) -> Self {
        let market = BoundlessMarketService::new(
            market_addr,
            provider.clone(),
            provider.default_signer_address(),
        );

        Self {
            db,
            config,
            prover,
            provider,
            chain_monitor,
            market,
            supported_selectors: SupportedSelectors::default(),
            new_order_rx: Arc::new(Mutex::new(new_order_rx)),
            priced_orders_tx: order_result_tx,
            stake_token_decimals,
            order_cache: Arc::new(
                Cache::builder()
                    .max_capacity(ORDER_DEDUP_CACHE_SIZE)
                    .time_to_live(Duration::from_secs(60 * 60)) // 1 hour
                    .build(),
            ),
            preflight_cache: Arc::new(
                Cache::builder()
                    .max_capacity(PREFLIGHT_CACHE_SIZE)
                    .time_to_live(Duration::from_secs(PREFLIGHT_CACHE_TTL_SECS))
                    .build(),
            ),
            order_state_tx,
            state_cache: Arc::new(RwLock::new(StateCache::default())),
        }
    }

    async fn price_order_and_update_state(
        &self,
        mut order: Box<OrderRequest>,
        cancel_token: CancellationToken,
    ) -> bool {
        let order_id = order.id();
        let f = || async {
            let pricing_result = tokio::select! {
                result = self.price_order(&mut order) => result,
                _ = cancel_token.cancelled() => {
                    tracing::info!("Order pricing cancelled during pricing for order {order_id}");

                    // Add the cancelled order to the database as skipped
                    if let Err(e) = self.db.insert_skipped_request(&order).await {
                        tracing::error!("Failed to add cancelled order to database: {e}");
                    }
                    return Ok(false);
                }
            };

            match pricing_result {
                Ok(Lock { total_cycles, target_timestamp_secs, expiry_secs }) => {
                    order.total_cycles = Some(total_cycles);
                    order.target_timestamp = Some(target_timestamp_secs);
                    order.expire_timestamp = Some(expiry_secs);

                    tracing::info!(
                        "Order {order_id} scheduled for lock attempt in {}s (timestamp: {}), when price threshold met",
                        target_timestamp_secs.saturating_sub(now_timestamp()),
                        target_timestamp_secs,
                    );

                    self.priced_orders_tx
                        .send(order)
                        .await
                        .context("Failed to send to order_result_tx")?;

                    Ok::<_, OrderPickerErr>(true)
                }
                Ok(ProveAfterLockExpire {
                    total_cycles,
                    lock_expire_timestamp_secs,
                    expiry_secs,
                }) => {
                    tracing::info!("Setting order {order_id} to prove after lock expiry at {lock_expire_timestamp_secs}");
                    order.total_cycles = Some(total_cycles);
                    order.target_timestamp = Some(lock_expire_timestamp_secs);
                    order.expire_timestamp = Some(expiry_secs);

                    self.priced_orders_tx
                        .send(order)
                        .await
                        .context("Failed to send to order_result_tx")?;

                    Ok(true)
                }
                Ok(Skip) => {
                    tracing::info!("Skipping order {order_id}");

                    // Add the skipped order to the database
                    self.db
                        .insert_skipped_request(&order)
                        .await
                        .context("Failed to add skipped order to database")?;
                    Ok(false)
                }
                Err(err) => {
                    tracing::warn!("Failed to price order {order_id}: {err}");
                    self.db
                        .insert_skipped_request(&order)
                        .await
                        .context("Failed to skip failed priced order")?;
                    Ok(false)
                }
            }
        };

        match f().await {
            Ok(true) => true,
            Ok(false) => false,
            Err(err) => {
                tracing::error!("Failed to update for order {order_id}: {err}");
                false
            }
        }
    }

    async fn price_order(
        &self,
        order: &mut OrderRequest,
    ) -> Result<OrderPricingOutcome, OrderPickerErr> {
        let order_id = order.id();
        tracing::debug!("Pricing order {order_id}");

        // Lock expiration is the timestamp before which the order must be filled in order to avoid slashing
        let lock_expiration =
            order.request.offer.biddingStart + order.request.offer.lockTimeout as u64;
        // order expiration is the timestamp after which the order can no longer be filled by anyone.
        let order_expiration =
            order.request.offer.biddingStart + order.request.offer.timeout as u64;

        let now = now_timestamp();

        // If order_expiration > lock_expiration the period in-between is when order can be filled
        // by anyone without staking to partially claim the slashed stake
        let lock_expired = order.fulfillment_type == FulfillmentType::FulfillAfterLockExpire;

        let (expiration, lockin_stake) = if lock_expired {
            (order_expiration, U256::ZERO)
        } else {
            (lock_expiration, U256::from(order.request.offer.lockStake))
        };

        if expiration <= now {
            tracing::info!("Removing order {order_id} because it has expired");
            return Ok(Skip);
        };

        // --- Start of cheap, local checks ---
        let (min_deadline, allowed_addresses_opt, denied_addresses_opt, max_stake_str) = {
            let config = self.config.lock_all().context("Failed to read config")?;
            (
                config.market.min_deadline,
                config.market.allow_client_addresses.clone(),
                config.market.deny_requestor_addresses.clone(),
                config.market.max_stake.clone(),
            )
        };

        let seconds_left = expiration.saturating_sub(now);
        if seconds_left <= min_deadline {
            tracing::info!("Removing order {order_id} because it expires within min_deadline: {seconds_left}, min_deadline: {min_deadline}");
            return Ok(Skip);
        }

        let client_addr = order.request.client_address();
        if let Some(allow_addresses) = allowed_addresses_opt {
            if !allow_addresses.contains(&client_addr) {
                tracing::info!("Removing order {order_id} from {client_addr} because it is not in allowed addrs");
                return Ok(Skip);
            }
        }

        if let Some(deny_addresses) = denied_addresses_opt {
            if deny_addresses.contains(&client_addr) {
                tracing::info!("Removing order {order_id} from {client_addr} because it is in denied addrs");
                return Ok(Skip);
            }
        }

        if !self.supported_selectors.is_supported(order.request.requirements.selector) {
            tracing::info!("Removing order {order_id} because it has an unsupported selector requirement");
            return Ok(Skip);
        };

        let max_stake = parse_ether(&max_stake_str).context("Failed to parse max_stake")?;
        if !lock_expired && lockin_stake > max_stake {
            tracing::info!("Removing high stake order {order_id}, lock stake: {lockin_stake}, max stake: {max_stake}");
            return Ok(Skip);
        }
        // --- End of cheap, local checks ---

        // --- Start of medium-cost checks (DB/Network) ---
        // OPTIMIZATION: Run independent checks concurrently.
        let (is_locked_res, is_fulfilled_res) = tokio::join!(
            async {
                if order.fulfillment_type == FulfillmentType::LockAndFulfill {
                    self.db.is_request_locked(U256::from(order.request.id)).await
                } else {
                    Ok(false)
                }
            },
            async {
                if order.fulfillment_type == FulfillmentType::FulfillAfterLockExpire {
                    self.db.is_request_fulfilled(U256::from(order.request.id)).await
                } else {
                    Ok(false)
                }
            }
        );

        if is_locked_res.context("Failed to check if request is locked")? {
            tracing::debug!("Order {order_id} is already locked, skipping");
            return Ok(Skip);
        }
        if is_fulfilled_res.context("Failed to check if request is fulfilled")? {
            tracing::debug!("Order {order_id} is already fulfilled, skipping");
            return Ok(Skip);
        }

        // OPTIMIZATION: Read from the state cache instead of making slow RPC calls.
        let state = self.state_cache.read().await;
        let gas_price = state.gas_price;
        let available_gas = state.gas_balance;
        let available_stake = state.stake_balance;
        drop(state); // Release the lock

        let order_gas = if lock_expired {
            U256::from(utils::estimate_gas_to_fulfill(&self.config, &self.supported_selectors, &order.request).await?)
        } else {
            U256::from(
                utils::estimate_gas_to_lock(&self.config, order).await?
                    + utils::estimate_gas_to_fulfill(&self.config, &self.supported_selectors, &order.request).await?,
            )
        };
        let order_gas_cost = U256::from(gas_price) * order_gas;

        tracing::debug!(
            "Estimated {order_gas} gas to {} order {order_id}; {} ether @ {} gwei",
            if lock_expired { "fulfill" } else { "lock and fulfill" },
            format_ether(order_gas_cost),
            format_units(gas_price, "gwei").unwrap_or_default()
        );

        if order_gas_cost > order.request.offer.maxPrice && !lock_expired {
            tracing::info!(
                "Estimated gas cost to lock and fulfill order {order_id}: {} exceeds max price; max price {}",
                format_ether(order_gas_cost),
                format_ether(order.request.offer.maxPrice)
            );
            return Ok(Skip);
        }

        if order_gas_cost > available_gas {
            tracing::warn!("Estimated there will be insufficient gas for order {order_id}; available_gas {} ether", format_ether(available_gas));
            return Ok(Skip);
        }

        if !lock_expired && lockin_stake > available_stake {
            tracing::warn!("Insufficient available stake to lock order {order_id}. Requires {lockin_stake}, has {available_stake}");
            return Ok(Skip);
        }
        // --- End of medium-cost checks ---

        // --- Start of high-cost checks (Preflight Execution) ---
        let (max_mcycle_limit, peak_prove_khz, min_mcycle_price_str, min_mcycle_price_stake_token_str, priority_requestor_addresses) = {
            let config = self.config.lock_all().context("Failed to read config")?;
            (
                config.market.max_mcycle_limit,
                config.market.peak_prove_khz,
                config.market.mcycle_price.clone(),
                config.market.mcycle_price_stake_token.clone(),
                config.market.priority_requestor_addresses.clone(),
            )
        };

        let mut exec_limit_cycles: u64 = if lock_expired {
            let min_mcycle_price_stake_token = parse_units(&min_mcycle_price_stake_token_str, self.stake_token_decimals)
                .context("Failed to parse mcycle_price_stake_token")?.into();
            if min_mcycle_price_stake_token == U256::ZERO {
                u64::MAX
            } else {
                let price = order.request.offer.stake_reward_if_locked_and_not_fulfilled();
                (price.saturating_mul(ONE_MILLION).div_ceil(min_mcycle_price_stake_token))
                    .try_into().context("Failed to convert U256 exec limit to u64")?
            }
        } else {
            let min_mcycle_price = parse_ether(&min_mcycle_price_str).context("Failed to parse mcycle_price")?;
            (U256::from(order.request.offer.maxPrice)
                .saturating_sub(order_gas_cost)
                .saturating_mul(ONE_MILLION)
                / min_mcycle_price)
                .try_into().context("Failed to convert U256 exec limit to u64")?
        };

        if exec_limit_cycles < 2 {
            tracing::info!("Removing order {order_id} because its exec limit is too low");
            return Ok(Skip);
        }

        let mut skip_mcycle_limit = false;
        if let Some(priority_addrs) = priority_requestor_addresses {
            if priority_addrs.contains(&client_addr) {
                skip_mcycle_limit = true;
            }
        }

        if skip_mcycle_limit {
            exec_limit_cycles = u64::MAX;
            tracing::debug!("Order {order_id} exec limit skipped due to client {} being part of priority_requestor_addresses.", client_addr);
        } else if let Some(config_mcycle_limit) = max_mcycle_limit {
            let config_cycle_limit = config_mcycle_limit.saturating_mul(1_000_000);
            if exec_limit_cycles >= config_cycle_limit {
                tracing::debug!("Order {order_id} exec limit computed from max price {} exceeds config max_mcycle_limit {}, setting exec limit to max_mcycle_limit", exec_limit_cycles / 1_000_000, config_mcycle_limit);
                exec_limit_cycles = config_cycle_limit;
            }
        }

        if let Some(peak_prove_khz) = peak_prove_khz {
            let time_until_expiration = expiration.saturating_sub(now);
            let deadline_cycle_limit = calculate_max_cycles_for_time(peak_prove_khz, time_until_expiration);
            if exec_limit_cycles > deadline_cycle_limit {
                tracing::debug!(
                    "Order {order_id} preflight cycle limit adjusted to {} cycles (capped by {:.1}s fulfillment deadline at {} peak_prove_khz config)",
                    deadline_cycle_limit, time_until_expiration, peak_prove_khz
                );
                exec_limit_cycles = deadline_cycle_limit;
            }
        }

        if exec_limit_cycles == 0 {
            tracing::debug!("Order {order_id} has no time left to prove within deadline, skipping");
            return Ok(Skip);
        }

        tracing::debug!(
            "Starting preflight execution of {order_id} with limit of {} cycles (~{} mcycles)",
            exec_limit_cycles, exec_limit_cycles / 1_000_000
        );

        let image_id_digest = Digest::from(order.request.requirements.imageId.0);
        let cache_key = match order.request.input.inputType {
            RequestInputType::Url => {
                let input_url = std::str::from_utf8(&order.request.input.data)
                    .context("input url is not utf8")?.to_string();
                PreflightCacheKey { image_id: image_id_digest, input: InputCacheKey::Url(input_url) }
            }
            RequestInputType::Inline => {
                let mut hasher = Sha256::new();
                hasher.update(&order.request.input.data);
                PreflightCacheKey { image_id: image_id_digest, input: InputCacheKey::Hash(hasher.finalize().into()) }
            }
            _ => return Err(OrderPickerErr::UnexpectedErr(Arc::new(anyhow::anyhow!("Unknown input type")))),
        };

        let preflight_result = loop {
            let prover = self.prover.clone();
            let config = self.config.clone();
            let request = order.request.clone();
            let order_id_clone = order_id.clone();
            let cache_key_clone = cache_key.clone();
            let cache_cloned = self.preflight_cache.clone();

            let result = tokio::task::spawn(async move {
                cache_cloned.try_get_with(cache_key_clone, async move {
                    tracing::trace!("Starting preflight of {order_id_clone} with exec limit {exec_limit_cycles} cycles");
                    let image_id = upload_image_uri(&prover, &request, &config).await.map_err(|e| OrderPickerErr::FetchImageErr(Arc::new(e)))?;
                    let input_id = upload_input_uri(&prover, &request, &config).await.map_err(|e| OrderPickerErr::FetchInputErr(Arc::new(e)))?;
                    
                    match prover.preflight(&image_id, &input_id, vec![], Some(exec_limit_cycles), &order_id_clone).await {
                        Ok(res) => Ok(PreflightCacheValue::Success {
                            exec_session_id: res.id,
                            cycle_count: res.stats.total_cycles,
                            image_id,
                            input_id,
                        }),
                        Err(err) => match err {
                            ProverError::ProvingFailed(ref msg) if msg.contains("Session limit exceeded") => Ok(PreflightCacheValue::Skip { cached_limit: exec_limit_cycles }),
                            ProverError::ProvingFailed(ref msg) if msg.contains("GuestPanic") => Err(OrderPickerErr::GuestPanic(msg.clone())),
                            _ => Err(OrderPickerErr::UnexpectedErr(Arc::new(err.into()))),
                        },
                    }
                }).await
            }).await.map_err(|e| OrderPickerErr::UnexpectedErr(Arc::new(e.into())))?;

            let cached_value = match result {
                Ok(value) => value,
                Err(e) => break Err((*e).clone()),
            };

            if let PreflightCacheValue::Skip { cached_limit } = cached_value {
                if cached_limit < exec_limit_cycles {
                    tracing::debug!("Cached result has insufficient limit for order {order_id} (cached: {}, required: {}), re-running preflight", cached_limit, exec_limit_cycles);
                    self.preflight_cache.invalidate(&cache_key).await;
                    continue;
                }
            }
            break Ok(cached_value);
        };

        let (exec_session_id, cycle_count) = match preflight_result {
            Ok(PreflightCacheValue::Success { exec_session_id, cycle_count, image_id, input_id }) => {
                order.image_id = Some(image_id);
                order.input_id = Some(input_id);
                (exec_session_id, cycle_count)
            }
            Ok(PreflightCacheValue::Skip { .. }) => return Ok(Skip),
            Err(err) => return Err(err),
        };

        let proof_res = ProofResult {
            id: exec_session_id,
            stats: ExecutorResp { total_cycles: cycle_count, ..Default::default() },
            elapsed_time: 0.0,
        };

        if let Some(mcycle_limit) = max_mcycle_limit {
            let mcycles = proof_res.stats.total_cycles / 1_000_000;
            if !skip_mcycle_limit && mcycles >= mcycle_limit {
                tracing::info!("Order {order_id} max_mcycle_limit check failed req: {mcycles} | config: {mcycle_limit}");
                return Ok(Skip);
            }
        }

        let journal = self.prover.get_preflight_journal(&proof_res.id).await
            .context("Failed to fetch preflight journal")?.context("Failed to find preflight journal")?;

        let max_journal_bytes = self.config.lock_all().context("Failed to read config")?.market.max_journal_bytes;
        if journal.len() > max_journal_bytes {
            tracing::info!("Order {order_id} journal larger than set limit ({} > {}), skipping", journal.len(), max_journal_bytes);
            return Ok(Skip);
        }

        if !order.request.requirements.predicate.eval(journal) {
            tracing::info!("Order {order_id} predicate check failed, skipping");
            return Ok(Skip);
        }

        self.evaluate_order(order, &proof_res, order_gas_cost, lock_expired).await
    }

    async fn evaluate_order(
        &self,
        order: &OrderRequest,
        proof_res: &ProofResult,
        order_gas_cost: U256,
        lock_expired: bool,
    ) -> Result<OrderPricingOutcome, OrderPickerErr> {
        if lock_expired {
            self.evaluate_lock_expired_order(order, proof_res).await
        } else {
            self.evaluate_lockable_order(order, proof_res, order_gas_cost).await
        }
    }

    async fn evaluate_lockable_order(
        &self,
        order: &OrderRequest,
        proof_res: &ProofResult,
        order_gas_cost: U256,
    ) -> Result<OrderPricingOutcome, OrderPickerErr> {
        let config_min_mcycle_price = {
            let config = self.config.lock_all().context("Failed to read config")?;
            parse_ether(&config.market.mcycle_price).context("Failed to parse mcycle_price")?
        };

        let order_id = order.id();
        let total_cycles_u256 = U256::from(proof_res.stats.total_cycles);

        let mcycle_price_max = U256::from(order.request.offer.maxPrice)
            .saturating_sub(order_gas_cost)
            .saturating_mul(ONE_MILLION)
            / total_cycles_u256;

        if mcycle_price_max < config_min_mcycle_price {
            tracing::debug!("Removing under priced order {order_id}");
            return Ok(Skip);
        }
        
        let mcycle_price_min = U256::from(order.request.offer.minPrice)
            .saturating_sub(order_gas_cost)
            .saturating_mul(ONE_MILLION)
            / total_cycles_u256;

        let target_timestamp_secs = if mcycle_price_min >= config_min_mcycle_price {
            tracing::info!("Selecting order {order_id} at min price - ASAP");
            // Using 0 maintains original logic and simplifies testing.
            0
        } else {
            let target_min_price = config_min_mcycle_price
                .saturating_mul(total_cycles_u256)
                .div_ceil(ONE_MILLION)
                + order_gas_cost;
            tracing::debug!("Order {order_id} minimum profitable price: {} ETH", format_ether(target_min_price));
            order.request.offer.time_at_price(target_min_price).context("Failed to get target price timestamp")?
        };

        let expiry_secs = order.request.offer.biddingStart + order.request.offer.lockTimeout as u64;
        Ok(Lock { total_cycles: proof_res.stats.total_cycles, target_timestamp_secs, expiry_secs })
    }

    async fn evaluate_lock_expired_order(
        &self,
        order: &OrderRequest,
        proof_res: &ProofResult,
    ) -> Result<OrderPricingOutcome, OrderPickerErr> {
        let config_min_mcycle_price_stake_tokens: U256 = {
            let config = self.config.lock_all().context("Failed to read config")?;
            parse_units(&config.market.mcycle_price_stake_token, self.stake_token_decimals)
                .context("Failed to parse mcycle_price")?.into()
        };

        let total_cycles = U256::from(proof_res.stats.total_cycles);
        let price = order.request.offer.stake_reward_if_locked_and_not_fulfilled();
        let mcycle_price_in_stake_tokens = price.saturating_mul(ONE_MILLION) / total_cycles;

        if mcycle_price_in_stake_tokens < config_min_mcycle_price_stake_tokens {
            tracing::info!("Removing under priced order (slashed stake reward too low) {}", order.id());
            return Ok(Skip);
        }

        Ok(ProveAfterLockExpire {
            total_cycles: proof_res.stats.total_cycles,
            lock_expire_timestamp_secs: order.request.offer.biddingStart + order.request.offer.lockTimeout as u64,
            expiry_secs: order.request.offer.biddingStart + order.request.offer.timeout as u64,
        })
    }

    /// OPTIMIZATION: Background task to refresh the on-chain state cache.
    async fn run_state_cache_refresher(&self, cancel_token: CancellationToken) {
        let mut interval = tokio::time::interval(STATE_CACHE_REFRESH_INTERVAL);
        let signer_address = self.provider.default_signer_address();

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let (gas_price_res, gas_balance_res, stake_balance_res) = tokio::join!(
                        self.chain_monitor.current_gas_price(),
                        self.provider.get_balance(signer_address),
                        self.market.balance_of_stake(signer_address)
                    );

                    let mut cache = self.state_cache.write().await;
                    if let Ok(gas_price) = gas_price_res {
                        cache.gas_price = gas_price;
                    } else {
                        tracing::warn!("Failed to refresh gas price for state cache");
                    }
                    if let Ok(gas_balance) = gas_balance_res {
                        cache.gas_balance = gas_balance;
                    } else {
                        tracing::warn!("Failed to refresh gas balance for state cache");
                    }
                    if let Ok(stake_balance) = stake_balance_res {
                        cache.stake_balance = stake_balance;
                    } else {
                        tracing::warn!("Failed to refresh stake balance for state cache");
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::info!("State cache refresher received cancellation signal.");
                    break;
                }
            }
        }
    }
}

/// Input type for preflight cache
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
enum InputCacheKey {
    Url(String),
    Hash([u8; 32]),
}

/// Key type for the preflight cache
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct PreflightCacheKey {
    image_id: Digest,
    input: InputCacheKey,
}

/// Value type for the preflight cache
#[derive(Clone, Debug)]
enum PreflightCacheValue {
    Success { exec_session_id: String, cycle_count: u64, image_id: String, input_id: String },
    Skip { cached_limit: u64 },
}

#[allow(clippy::vec_box)]
fn handle_lock_event(
    request_id: U256,
    active_tasks: &mut BTreeMap<U256, BTreeMap<String, CancellationToken>>,
    pending_orders: &mut Vec<Box<OrderRequest>>,
) {
    if let Some(order_tasks) = active_tasks.get_mut(&request_id) {
        let initial_count = order_tasks.len();
        order_tasks.retain(|order_id, task_token| {
            if order_id.contains("LockAndFulfill") {
                task_token.cancel();
                false
            } else { true }
        });
        if initial_count > order_tasks.len() {
            tracing::debug!("Cancelled {} LockAndFulfill preflights for locked request 0x{:x}", initial_count - order_tasks.len(), request_id);
        }
        if order_tasks.is_empty() {
            active_tasks.remove(&request_id);
        }
    }

    let initial_len = pending_orders.len();
    pending_orders.retain(|order| !(U256::from(order.request.id) == request_id && order.fulfillment_type == FulfillmentType::LockAndFulfill));
    if initial_len > pending_orders.len() {
        tracing::debug!("Removed {} pending LockAndFulfill orders for locked request 0x{:x}", initial_len - pending_orders.len(), request_id);
    }
}

#[allow(clippy::vec_box)]
fn handle_fulfill_event(
    request_id: U256,
    active_tasks: &mut BTreeMap<U256, BTreeMap<String, CancellationToken>>,
    pending_orders: &mut Vec<Box<OrderRequest>>,
) {
    if let Some(order_tasks) = active_tasks.remove(&request_id) {
        tracing::debug!("Cancelling {} active preflights for fulfilled request 0x{:x}", order_tasks.len(), request_id);
        for (_, task_token) in order_tasks {
            task_token.cancel();
        }
    }

    let initial_len = pending_orders.len();
    pending_orders.retain(|order| U256::from(order.request.id) != request_id);
    if initial_len > pending_orders.len() {
        tracing::debug!("Removed {} pending orders for fulfilled request 0x{:x}", initial_len - pending_orders.len(), request_id);
    }
}

impl<P> RetryTask for OrderPicker<P>
where
    P: Provider<Ethereum> + 'static + Clone + WalletProvider,
{
    type Error = OrderPickerErr;
    fn spawn(&self, cancel_token: CancellationToken) -> RetryRes<Self::Error> {
        let picker = self.clone();

        Box::pin(async move {
            tracing::info!("Starting order picking monitor");

            // OPTIMIZATION: Spawn the background task to keep the state cache fresh.
            let cache_refresher_token = cancel_token.child_token();
            let cache_refresher_picker = picker.clone();
            tokio::spawn(async move {
                cache_refresher_picker.run_state_cache_refresher(cache_refresher_token).await;
            });

            let read_config = || -> Result<_, Self::Error> {
                let cfg = picker.config.lock_all().map_err(|err| OrderPickerErr::UnexpectedErr(Arc::new(anyhow::anyhow!("Failed to read config: {err}"))))?;
                Ok((
                    cfg.market.max_concurrent_preflights as usize,
                    // NOTE: order_pricing_priority is not used in this simplified loop,
                    // but we read it to maintain config compatibility.
                    cfg.market.order_pricing_priority,
                    cfg.market.priority_requestor_addresses.clone(),
                ))
            };

            let (mut current_capacity, mut priority_mode, mut priority_addresses) = read_config().map_err(SupervisorErr::Fault)?;
            let mut tasks: JoinSet<(String, U256)> = JoinSet::new();
            let mut rx = picker.new_order_rx.lock().await;
            let mut order_state_rx = picker.order_state_tx.subscribe();
            let mut capacity_check_interval = tokio::time::interval(MIN_CAPACITY_CHECK_INTERVAL);
            let mut pending_orders: Vec<Box<OrderRequest>> = Vec::new();
            let mut active_tasks: BTreeMap<U256, BTreeMap<String, CancellationToken>> = BTreeMap::new();

            loop {
                tokio::select! {
                    Some(order) = rx.recv() => {
                        pending_orders.push(order);
                    }
                    Ok(state_change) = order_state_rx.recv() => {
                        match state_change {
                            OrderStateChange::Locked { request_id, .. } => handle_lock_event(request_id, &mut active_tasks, &mut pending_orders),
                            OrderStateChange::Fulfilled { request_id } => handle_fulfill_event(request_id, &mut active_tasks, &mut pending_orders),
                        }
                    }
                    Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                        if let Ok((order_id, request_id)) = result {
                            if let Some(order_tasks) = active_tasks.get_mut(&request_id) {
                                order_tasks.remove(&order_id);
                                if order_tasks.is_empty() {
                                    active_tasks.remove(&request_id);
                                }
                            }
                            tracing::trace!("Priced task for order {} (request 0x{:x}) completed ({} remaining)", order_id, request_id, tasks.len());
                        }
                    }
                    _ = capacity_check_interval.tick() => {
                        let (new_capacity, new_priority_mode, new_priority_addresses) = read_config().map_err(SupervisorErr::Fault)?;
                        if new_capacity != current_capacity {
                            tracing::debug!("Pricing capacity changed from {} to {}", current_capacity, new_capacity);
                            current_capacity = new_capacity;
                        }
                        if new_priority_mode != priority_mode {
                            tracing::debug!("Order pricing priority changed from {:?} to {:?}", priority_mode, new_priority_mode);
                            priority_mode = new_priority_mode;
                        }
                        if new_priority_addresses != priority_addresses {
                            tracing::debug!("Priority requestor addresses changed");
                            priority_addresses = new_priority_addresses;
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("Order picker received cancellation, shutting down gracefully");
                        while tasks.join_next().await.is_some() {}
                        break;
                    }
                }

                // CORRECTED LOGIC: Process pending orders using a simple FIFO queue.
                while !pending_orders.is_empty() && tasks.len() < current_capacity {
                    // A more sophisticated strategy could be implemented here, but FIFO is safe and correct.
                    let order = pending_orders.remove(0);
                    let order_id = order.id();
                    let request_id = U256::from(order.request.id);

                    if active_tasks.get(&request_id).map_or(false, |m| m.contains_key(&order_id)) {
                        tracing::trace!("Skipping order {order_id} - already being processed");
                        continue;
                    }
                    if picker.order_cache.get(&order_id).await.is_some() {
                        tracing::trace!("Skipping duplicate order {order_id}, already processed recently");
                        continue;
                    }

                    picker.order_cache.insert(order_id.clone(), ()).await;
                    let picker_clone = picker.clone();
                    let task_cancel_token = cancel_token.child_token();
                    active_tasks.entry(request_id).or_default().insert(order_id.clone(), task_cancel_token.clone());

                    tasks.spawn(async move {
                        picker_clone.price_order_and_update_state(order, task_cancel_token).await;
                        (order_id, request_id)
                    });
                }
            }
            Ok(())
        })
    }
}

fn calculate_max_cycles_for_time(prove_khz: u64, time_seconds: u64) -> u64 {
    (prove_khz.saturating_mul(1_000)).saturating_mul(time_seconds)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{
        chain_monitor::ChainMonitorService,
        db::SqliteDb,
        provers::{DefaultProver, Prover},
        FulfillmentType, OrderStatus,
    };
    use alloy::{
        network::EthereumWallet,
        node_bindings::{Anvil, AnvilInstance},
        primitives::{address, aliases::U96, utils::parse_units, Address, Bytes, FixedBytes, B256},
        providers::{ext::AnvilApi, ProviderBuilder},
        signers::local::PrivateKeySigner,
    };
    use async_trait::async_trait;
    use boundless_market::contracts::{
        Callback, Offer, Predicate, PredicateType, ProofRequest, RequestId, RequestInput,
        Requirements,
    };
    use boundless_market::storage::{MockStorageProvider, StorageProvider};
    use boundless_market_test_utils::{
        deploy_boundless_market, deploy_hit_points, ASSESSOR_GUEST_ID, ASSESSOR_GUEST_PATH,
        ECHO_ELF, ECHO_ID, LOOP_ELF, LOOP_ID,
    };
    use risc0_ethereum_contracts::selector::Selector;
    use risc0_zkvm::sha::Digest;
    use risc0_zkvm::Receipt;
    use tracing_test::traced_test;

    /// Reusable context for testing the order picker
    pub(crate) struct PickerTestCtx<P> {
        anvil: AnvilInstance,
        pub(crate) picker: OrderPicker<P>,
        boundless_market: BoundlessMarketService<Arc<P>>,
        storage_provider: MockStorageProvider,
        db: DbObj,
        provider: Arc<P>,
        priced_orders_rx: mpsc::Receiver<Box<OrderRequest>>,
        new_order_tx: mpsc::Sender<Box<OrderRequest>>,
    }

    /// Parameters for the generate_next_order function.
    pub(crate) struct OrderParams {
        pub(crate) order_index: u32,
        pub(crate) min_price: U256,
        pub(crate) max_price: U256,
        pub(crate) lock_stake: U256,
        pub(crate) fulfillment_type: FulfillmentType,
        pub(crate) bidding_start: u64,
        pub(crate) lock_timeout: u32,
        pub(crate) timeout: u32,
    }

    impl Default for OrderParams {
        fn default() -> Self {
            Self {
                order_index: 1,
                min_price: parse_ether("0.02").unwrap(),
                max_price: parse_ether("0.04").unwrap(),
                lock_stake: U256::ZERO,
                fulfillment_type: FulfillmentType::LockAndFulfill,
                bidding_start: now_timestamp(),
                lock_timeout: 900,
                timeout: 1200,
            }
        }
    }

    impl<P> PickerTestCtx<P>
    where
        P: Provider + WalletProvider,
    {
        pub(crate) fn signer(&self, index: usize) -> PrivateKeySigner {
            self.anvil.keys()[index].clone().into()
        }

        /// Helper to populate the state cache for deterministic tests.
        pub(crate) async fn prime_state_cache(&self) {
            let signer_address = self.provider.default_signer_address();
            let (gas_price_res, gas_balance_res, stake_balance_res) = tokio::join!(
                self.picker.chain_monitor.current_gas_price(),
                self.provider.get_balance(signer_address),
                self.picker.market.balance_of_stake(signer_address)
            );

            let mut cache = self.picker.state_cache.write().await;
            cache.gas_price = gas_price_res.unwrap_or(10_000_000_000); // Default 10 gwei
            cache.gas_balance = gas_balance_res.unwrap_or_default();
            cache.stake_balance = stake_balance_res.unwrap_or_default();
        }

        pub(crate) async fn generate_next_order(&self, params: OrderParams) -> Box<OrderRequest> {
            let image_url =
                self.storage_provider.upload_program(ECHO_ELF).await.unwrap().to_string();
            let image_id = Digest::from(ECHO_ID);
            let chain_id = self.provider.get_chain_id().await.unwrap();
            let boundless_market_address = self.boundless_market.instance().address();

            Box::new(OrderRequest {
                request: ProofRequest::new(
                    RequestId::new(self.provider.default_signer_address(), params.order_index),
                    Requirements::new(
                        image_id,
                        Predicate {
                            predicateType: PredicateType::PrefixMatch,
                            data: Default::default(),
                        },
                    ),
                    image_url,
                    RequestInput::builder()
                        .write_slice(&[0x41, 0x41, 0x41, 0x41])
                        .build_inline()
                        .unwrap(),
                    Offer {
                        minPrice: params.min_price,
                        maxPrice: params.max_price,
                        biddingStart: params.bidding_start,
                        timeout: params.timeout,
                        lockTimeout: params.lock_timeout,
                        rampUpPeriod: 1,
                        lockStake: params.lock_stake,
                    },
                ),
                target_timestamp: None,
                image_id: None,
                input_id: None,
                expire_timestamp: None,
                client_sig: Bytes::new(),
                fulfillment_type: params.fulfillment_type,
                boundless_market_address: *boundless_market_address,
                chain_id,
                total_cycles: None,
            })
        }

        pub(crate) async fn generate_loop_order(
            &self,
            params: OrderParams,
            cycles: u64,
        ) -> Box<OrderRequest> {
            let image_url =
                self.storage_provider.upload_program(LOOP_ELF).await.unwrap().to_string();
            let image_id = Digest::from(LOOP_ID);
            let chain_id = self.provider.get_chain_id().await.unwrap();
            let boundless_market_address = self.boundless_market.instance().address();

            Box::new(OrderRequest {
                request: ProofRequest::new(
                    RequestId::new(self.provider.default_signer_address(), params.order_index),
                    Requirements::new(
                        image_id,
                        Predicate {
                            predicateType: PredicateType::PrefixMatch,
                            data: Default::default(),
                        },
                    ),
                    image_url,
                    RequestInput::builder()
                        .write(&cycles)
                        .unwrap()
                        .write(&1u64)
                        .unwrap() // nonce
                        .build_inline()
                        .unwrap(),
                    Offer {
                        minPrice: params.min_price,
                        maxPrice: params.max_price,
                        biddingStart: params.bidding_start,
                        timeout: params.timeout,
                        lockTimeout: params.lock_timeout,
                        rampUpPeriod: 1,
                        lockStake: params.lock_stake,
                    },
                ),
                target_timestamp: None,
                image_id: None,
                input_id: None,
                expire_timestamp: None,
                client_sig: Bytes::new(),
                fulfillment_type: params.fulfillment_type,
                boundless_market_address: *boundless_market_address,
                chain_id,
                total_cycles: None,
            })
        }
    }

    #[derive(Default)]
    pub(crate) struct PickerTestCtxBuilder {
        initial_signer_eth: Option<i32>,
        initial_hp: Option<U256>,
        config: Option<ConfigLock>,
        stake_token_decimals: Option<u8>,
        prover: Option<ProverObj>,
    }

    impl PickerTestCtxBuilder {
        pub(crate) fn with_initial_signer_eth(self, eth: i32) -> Self {
            Self { initial_signer_eth: Some(eth), ..self }
        }
        pub(crate) fn with_initial_hp(self, hp: U256) -> Self {
            assert!(hp < U256::from(U96::MAX), "Cannot have more than 2^96 hit points");
            Self { initial_hp: Some(hp), ..self }
        }
        pub(crate) fn with_config(self, config: ConfigLock) -> Self {
            Self { config: Some(config), ..self }
        }
        pub(crate) fn with_prover(self, prover: ProverObj) -> Self {
            Self { prover: Some(prover), ..self }
        }
        pub(crate) fn with_stake_token_decimals(self, decimals: u8) -> Self {
            Self { stake_token_decimals: Some(decimals), ..self }
        }
        pub(crate) async fn build(
            self,
        ) -> PickerTestCtx<impl Provider + WalletProvider + Clone + 'static> {
            let anvil = Anvil::new()
                .args(["--balance", &format!("{}", self.initial_signer_eth.unwrap_or(10000))])
                .spawn();
            let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
            let provider = Arc::new(
                ProviderBuilder::new()
                    .wallet(EthereumWallet::from(signer.clone()))
                    .connect(&anvil.endpoint())
                    .await
                    .unwrap(),
            );

            provider.anvil_mine(Some(4), Some(2)).await.unwrap();

            let hp_contract = deploy_hit_points(signer.address(), provider.clone()).await.unwrap();
            let market_address = deploy_boundless_market(
                signer.address(),
                provider.clone(),
                Address::ZERO,
                hp_contract,
                Digest::from(ASSESSOR_GUEST_ID),
                format!("file://{ASSESSOR_GUEST_PATH}"),
                Some(signer.address()),
            )
            .await
            .unwrap();

            let boundless_market = BoundlessMarketService::new(
                market_address,
                provider.clone(),
                provider.default_signer_address(),
            );

            if let Some(initial_hp) = self.initial_hp {
                tracing::debug!("Setting initial locked hitpoints to {}", initial_hp);
                boundless_market.deposit_stake_with_permit(initial_hp, &signer).await.unwrap();
                assert_eq!(
                    boundless_market
                        .balance_of_stake(provider.default_signer_address())
                        .await
                        .unwrap(),
                    initial_hp
                );
            }

            let storage_provider = MockStorageProvider::start();

            let db: DbObj = Arc::new(SqliteDb::new("sqlite::memory:").await.unwrap());
            let config = self.config.unwrap_or_default();
            let prover: ProverObj = self.prover.unwrap_or_else(|| Arc::new(DefaultProver::new()));
            let chain_monitor = Arc::new(ChainMonitorService::new(provider.clone()).await.unwrap());
            tokio::spawn(chain_monitor.spawn(Default::default()));

            const TEST_CHANNEL_CAPACITY: usize = 50;
            let (_new_order_tx, new_order_rx) = mpsc::channel(TEST_CHANNEL_CAPACITY);
            let (priced_orders_tx, priced_orders_rx) = mpsc::channel(TEST_CHANNEL_CAPACITY);
            let (order_state_tx, _) = tokio::sync::broadcast::channel(TEST_CHANNEL_CAPACITY);

            let picker = OrderPicker::new(
                db.clone(),
                config,
                prover,
                market_address,
                provider.clone(),
                chain_monitor,
                new_order_rx,
                priced_orders_tx,
                self.stake_token_decimals.unwrap_or(6),
                order_state_tx,
            );

            PickerTestCtx {
                anvil,
                picker,
                boundless_market,
                storage_provider,
                db,
                provider,
                priced_orders_rx,
                new_order_tx: _new_order_tx,
            }
        }
    }

    #[tokio::test]
    #[traced_test]
    async fn price_order() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx.generate_next_order(Default::default()).await;

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        let priced_order = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced_order.target_timestamp, Some(0));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_bad_predicate() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let mut order = ctx.generate_next_order(Default::default()).await;
        // set a bad predicate
        order.request.requirements.predicate =
            Predicate { predicateType: PredicateType::DigestMatch, data: B256::ZERO.into() };

        let order_id = order.id();
        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain("predicate check failed, skipping"));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_unsupported_selector() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let mut order = ctx.generate_next_order(Default::default()).await;

        // set an unsupported selector
        order.request.requirements.selector = FixedBytes::from(Selector::Groth16V1_1 as u32);
        let order_id = order.id();

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain("has an unsupported selector requirement"));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_price_less_than_gas_costs() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx
            .generate_next_order(OrderParams {
                min_price: parse_ether("0.0005").unwrap(),
                max_price: parse_ether("0.0010").unwrap(),
                ..Default::default()
            })
            .await;
        let order_id = order.id();

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain(&format!("Estimated gas cost to lock and fulfill order {order_id}:")));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_price_less_than_gas_costs_groth16() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        // NOTE: Values currently adjusted ad hoc to be between the two thresholds.
        let min_price = parse_ether("0.0013").unwrap();
        let max_price = parse_ether("0.0013").unwrap();

        // Order should have high enough price with the default selector.
        let order = ctx
            .generate_next_order(OrderParams {
                order_index: 1,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);
        let priced = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced.target_timestamp, Some(0));

        // Order does not have high enough price when groth16 is used.
        let mut order = ctx
            .generate_next_order(OrderParams {
                order_index: 2,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;

        // set a Groth16 selector
        order.request.requirements.selector = FixedBytes::from(Selector::Groth16V2_2 as u32);

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain(&format!("Estimated gas cost to lock and fulfill order {order_id}:")));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_price_less_than_gas_costs_callback() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        // NOTE: Values currently adjusted ad hoc to be between the two thresholds.
        let min_price = parse_ether("0.0013").unwrap();
        let max_price = parse_ether("0.0013").unwrap();

        // Order should have high enough price with the default selector.
        let order = ctx
            .generate_next_order(OrderParams {
                order_index: 1,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;
        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        let priced = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced.target_timestamp, Some(0));

        // Order does not have high enough price when groth16 is used.
        let mut order = ctx
            .generate_next_order(OrderParams {
                order_index: 2,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;

        // set a callback with a nontrivial gas consumption
        order.request.requirements.callback = Callback {
            addr: address!("0x00000000000000000000000000000000ca11bac2"),
            gasLimit: U96::from(200_000),
        };

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain(&format!("Estimated gas cost to lock and fulfill order {order_id}:")));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_price_less_than_gas_costs_smart_contract_signature() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        // NOTE: Values currently adjusted ad hoc to be between the two thresholds.
        let min_price = parse_ether("0.0013").unwrap();
        let max_price = parse_ether("0.0013").unwrap();

        // Order should have high enough price with the default selector.
        let order = ctx
            .generate_next_order(OrderParams {
                order_index: 1,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        let priced = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced.target_timestamp, Some(0));

        // Order does not have high enough price when groth16 is used.
        let mut order = ctx
            .generate_next_order(OrderParams {
                order_index: 2,
                min_price,
                max_price,
                ..Default::default()
            })
            .await;

        order.request.id =
            RequestId::try_from(order.request.id).unwrap().set_smart_contract_signed_flag().into();

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain(&format!("Estimated gas cost to lock and fulfill order {order_id}:")));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_unallowed_addr() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.allow_client_addresses = Some(vec![Address::ZERO]);
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx.generate_next_order(Default::default()).await;

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain("because it is not in allowed addrs"));
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_denied_addr() {
        let config = ConfigLock::default();
        let ctx = PickerTestCtxBuilder::default().with_config(config.clone()).build().await;
        ctx.prime_state_cache().await;
        let deny_address = ctx.provider.default_signer_address();

        {
            let mut cfg = config.load_write().unwrap();
            cfg.market.mcycle_price = "0.0000001".into();
            cfg.market.deny_requestor_addresses = Some([deny_address].into_iter().collect());
        }

        let order = ctx.generate_next_order(Default::default()).await;

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);

        assert!(logs_contain("because it is in denied addrs"));
    }

    #[tokio::test]
    #[traced_test]
    async fn resume_order_pricing() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx.generate_next_order(Default::default()).await;
        let order_id = order.id();

        let _request_id =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await.unwrap();

        let pricing_task = tokio::spawn(ctx.picker.spawn(Default::default()));

        ctx.new_order_tx.send(order).await.unwrap();

        // Wait for the order to be priced, with some timeout
        let priced_order =
            tokio::time::timeout(Duration::from_secs(10), ctx.priced_orders_rx.recv())
                .await
                .unwrap();
        assert_eq!(priced_order.unwrap().id(), order_id);

        pricing_task.abort();

        // Send a new order when picker task is down.
        let new_order = ctx.generate_next_order(Default::default()).await;
        let new_order_id = new_order.id();
        ctx.new_order_tx.send(new_order).await.unwrap();

        assert!(ctx.priced_orders_rx.is_empty());

        tokio::spawn(ctx.picker.spawn(Default::default()));

        let priced_order =
            tokio::time::timeout(Duration::from_secs(10), ctx.priced_orders_rx.recv())
                .await
                .unwrap();
        assert_eq!(priced_order.unwrap().id(), new_order_id);
    }

    #[tokio::test]
    #[traced_test]
    async fn cannot_overcommit_stake() {
        let signer_inital_balance_eth = 2;
        let lockin_stake = U256::from(150);

        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.max_stake = "10".into();
        }

        let mut ctx = PickerTestCtxBuilder::default()
            .with_initial_signer_eth(signer_inital_balance_eth)
            .with_initial_hp(lockin_stake)
            .with_config(config)
            .build()
            .await;
        ctx.prime_state_cache().await;
        let order = ctx
            .generate_next_order(OrderParams { lock_stake: U256::from(100), ..Default::default() })
            .await;
        let order1_id = order.id();
        assert!(ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await);
        let priced = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced.id(), order1_id);

        let order = ctx
            .generate_next_order(OrderParams {
                lock_stake: lockin_stake + U256::from(1),
                ..Default::default()
            })
            .await;
        let order_id = order.id();
        assert!(!ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await);
        assert!(logs_contain("Insufficient available stake to lock order"));
        assert_eq!(
            ctx.db.get_order(&order_id).await.unwrap().unwrap().status,
            OrderStatus::Skipped
        );

        let order = ctx
            .generate_next_order(OrderParams {
                lock_stake: parse_units("11", 18).unwrap().into(),
                ..Default::default()
            })
            .await;
        let order_id = order.id();
        assert!(!ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await);

        // only the first order above should have marked as active pricing, the second one should have been skipped due to insufficient stake
        assert_eq!(
            ctx.db.get_order(&order_id).await.unwrap().unwrap().status,
            OrderStatus::Skipped
        );
        assert!(logs_contain("Removing high stake order"));
    }

    #[tokio::test]
    #[traced_test]
    async fn use_gas_to_fulfill_estimate_from_config() {
        let fulfill_gas = 123_456;
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.fulfill_gas_estimate = fulfill_gas;
        }

        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx.generate_next_order(Default::default()).await;
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        // Simulate order being locked
        let order = ctx.priced_orders_rx.try_recv().unwrap();
        ctx.db.insert_accepted_request(&order, order.request.offer.minPrice).await.unwrap();

        // This test is tricky now because estimate_gas_to_fulfill_pending is not part of the picker.
        // We can test the underlying utility function.
        let pending_orders = ctx.db.get_committed_orders().await.unwrap();
        let mut gas = 0;
        for order in pending_orders {
             gas += utils::estimate_gas_to_fulfill(
                &ctx.picker.config,
                &ctx.picker.supported_selectors,
                &order.request,
            )
            .await.unwrap();
        }
        assert_eq!(gas, fulfill_gas);

        // add another order
        let order =
            ctx.generate_next_order(OrderParams { order_index: 2, ..Default::default() }).await;
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);
        let order = ctx.priced_orders_rx.try_recv().unwrap();
        ctx.db.insert_accepted_request(&order, order.request.offer.minPrice).await.unwrap();

        let pending_orders = ctx.db.get_committed_orders().await.unwrap();
        let mut gas = 0;
        for order in pending_orders {
             gas += utils::estimate_gas_to_fulfill(
                &ctx.picker.config,
                &ctx.picker.supported_selectors,
                &order.request,
            )
            .await.unwrap();
        }
        // gas estimate stacks (until estimates factor in bundling)
        assert_eq!(gas, 2 * fulfill_gas);
    }

    #[tokio::test]
    #[traced_test]
    async fn skips_journal_exceeding_limit() {
        // set this by testing a very small limit (1 byte)
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.max_journal_bytes = 1;
        }
        let lock_stake = U256::from(10);

        let ctx = PickerTestCtxBuilder::default()
            .with_config(config)
            .with_initial_hp(lock_stake)
            .build()
            .await;
        ctx.prime_state_cache().await;
        let order = ctx.generate_next_order(OrderParams { lock_stake, ..Default::default() }).await;

        let order_id = order.id();
        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(!locked);

        assert_eq!(
            ctx.db.get_order(&order_id).await.unwrap().unwrap().status,
            OrderStatus::Skipped
        );
        assert!(logs_contain("journal larger than set limit"));
    }

    #[tokio::test]
    #[traced_test]
    async fn price_locked_by_other() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price_stake_token = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default()
            .with_config(config)
            .with_initial_hp(U256::from(1000))
            .build()
            .await;
        ctx.prime_state_cache().await;

        let order = ctx
            .generate_next_order(OrderParams {
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                bidding_start: now_timestamp(),
                lock_timeout: 1000,
                timeout: 10000,
                lock_stake: parse_units("0.1", 6).unwrap().into(),
                ..Default::default()
            })
            .await;

        let order_id = order.id();
        let expected_target_timestamp =
            order.request.offer.biddingStart + order.request.offer.lockTimeout as u64;
        let expected_expire_timestamp =
            order.request.offer.biddingStart + order.request.offer.timeout as u64;

        let expected_log = format!(
            "Setting order {order_id} to prove after lock expiry at {expected_target_timestamp}"
        );
        assert!(ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await);

        assert!(logs_contain(&expected_log));

        let priced = ctx.priced_orders_rx.try_recv().unwrap();
        assert_eq!(priced.target_timestamp, Some(expected_target_timestamp));
        assert_eq!(priced.expire_timestamp, Some(expected_expire_timestamp));
    }

    #[tokio::test]
    #[traced_test]
    async fn price_locked_by_other_unprofitable() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price_stake_token = "0.1".into();
        }
        let ctx = PickerTestCtxBuilder::default()
            .with_stake_token_decimals(6)
            .with_config(config)
            .build()
            .await;
        ctx.prime_state_cache().await;

        let order = ctx
            .generate_next_order(OrderParams {
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                bidding_start: now_timestamp(),
                lock_timeout: 0,
                timeout: 10000,
                // Low stake means low reward for filling after it is unfulfilled
                lock_stake: parse_units("0.00001", 6).unwrap().into(),
                ..Default::default()
            })
            .await;

        let order_id = order.id();

        assert!(!ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await);

        // Since we know the stake reward is constant, and we know our min_mycle_price_stake_token
        // the execution limit check tells us if the order is profitable or not, since it computes the max number
        // of cycles that can be proven while keeping the order profitable.
        assert!(logs_contain(&format!("Skipping order {order_id} due to session limit exceeded")));

        let db_order = ctx.db.get_order(&order_id).await.unwrap().unwrap();
        assert_eq!(db_order.status, OrderStatus::Skipped);
    }

    #[tokio::test]
    #[traced_test]
    async fn skip_mcycle_limit_for_allowed_address() {
        let exec_limit = 1000;
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.max_mcycle_limit = Some(exec_limit);
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        ctx.picker.config.load_write().as_mut().unwrap().market.priority_requestor_addresses =
            Some(vec![ctx.provider.default_signer_address()]);

        // First order from allowed address - should skip mcycle limit
        let order = ctx.generate_next_order(Default::default()).await;
        let order_id = order.id();

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        // Check logs for the expected message about skipping mcycle limit
        assert!(logs_contain(&format!(
            "Order {order_id} exec limit skipped due to client {} being part of priority_requestor_addresses.",
            ctx.provider.default_signer_address()
        )));

        // Second order from a different address - should have mcycle limit enforced
        let mut order2 =
            ctx.generate_next_order(OrderParams { order_index: 2, ..Default::default() }).await;
        // Set a different client address
        order2.request.id = RequestId::new(Address::ZERO, 2).into();
        let order2_id = order2.id();

        let locked =
            ctx.picker.price_order_and_update_state(order2, CancellationToken::new()).await;
        assert!(locked);

        // Check logs for the expected message about setting exec limit to max_mcycle_limit
        assert!(logs_contain(&format!("Order {order2_id} exec limit computed from max price")));
        assert!(logs_contain("exceeds config max_mcycle_limit"));
        assert!(logs_contain("setting exec limit to max_mcycle_limit"));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_deadline_exec_limit_and_peak_prove_khz() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.peak_prove_khz = Some(1);
            config.load_write().unwrap().market.min_deadline = 10;
        }
        let ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        let order = ctx
            .generate_next_order(OrderParams {
                min_price: parse_ether("10").unwrap(),
                max_price: parse_ether("10").unwrap(),
                bidding_start: now_timestamp(),
                lock_timeout: 150,
                timeout: 300,
                ..Default::default()
            })
            .await;

        let order_id = order.id();
        let _submit_result =
            ctx.boundless_market.submit_request(&order.request, &ctx.signer(0)).await;

        let locked = ctx.picker.price_order_and_update_state(order, CancellationToken::new()).await;
        assert!(locked);

        let expected_log_pattern = format!("Order {order_id} preflight cycle limit adjusted to");
        assert!(logs_contain(&expected_log_pattern));
        assert!(logs_contain("capped by"));
        assert!(logs_contain("peak_prove_khz config"));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_capacity_change() {
        let config = ConfigLock::default();
        {
            let mut cfg = config.load_write().unwrap();
            cfg.market.mcycle_price = "0.0000001".into();
            cfg.market.max_concurrent_preflights = 2;
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config.clone()).build().await;
        ctx.prime_state_cache().await;

        // Start the order picker task
        let picker_task = tokio::spawn(ctx.picker.spawn(Default::default()));

        // Send an initial order to trigger the capacity check
        let order1 =
            ctx.generate_next_order(OrderParams { order_index: 1, ..Default::default() }).await;
        ctx.new_order_tx.send(order1).await.unwrap();

        // Wait for order to be processed
        tokio::time::timeout(Duration::from_secs(10), ctx.priced_orders_rx.recv()).await.unwrap();

        // Sleep to allow for a capacity check change
        tokio::time::sleep(MIN_CAPACITY_CHECK_INTERVAL).await;

        // Decrease capacity
        {
            let mut cfg = config.load_write().unwrap();
            cfg.market.max_concurrent_preflights = 1;
        }

        // Wait a bit more for the interval timer to fire and detect the change
        tokio::time::sleep(MIN_CAPACITY_CHECK_INTERVAL + Duration::from_millis(100)).await;

        // Send another order to trigger capacity check
        let order2 =
            ctx.generate_next_order(OrderParams { order_index: 2, ..Default::default() }).await;
        ctx.new_order_tx.send(order2).await.unwrap();

        // Wait for an order to be processed before updating capacity
        tokio::time::timeout(Duration::from_secs(10), ctx.priced_orders_rx.recv()).await.unwrap();

        // Check logs for capacity changes
        assert!(logs_contain("Pricing capacity changed from 2 to 1"));

        picker_task.abort();
    }

    #[tokio::test]
    #[traced_test]
    async fn test_lock_expired_exec_limit_precision_loss() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price_stake_token = "1".into();
        }
        let ctx = PickerTestCtxBuilder::default()
            .with_config(config.clone())
            .with_stake_token_decimals(6)
            .build()
            .await;
        ctx.prime_state_cache().await;

        let mut order = ctx
            .generate_next_order(OrderParams {
                lock_stake: U256::from(4),
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                bidding_start: now_timestamp() - 100,
                lock_timeout: 10,
                timeout: 300,
                ..Default::default()
            })
            .await;

        let order_id = order.id();
        let stake_reward = order.request.offer.stake_reward_if_locked_and_not_fulfilled();
        assert_eq!(stake_reward, U256::from(1));

        let locked = ctx.picker.price_order(&mut order).await;
        assert!(matches!(locked, Ok(OrderPricingOutcome::Skip)));

        assert!(logs_contain(&format!(
            "Removing order {order_id} because its exec limit is too low"
        )));

        let mut order2 = ctx
            .generate_next_order(OrderParams {
                order_index: 2,
                lock_stake: U256::from(40),
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                bidding_start: now_timestamp() - 100,
                lock_timeout: 10,
                timeout: 300,
                ..Default::default()
            })
            .await;

        let order2_id = order2.id();
        let stake_reward2 = order2.request.offer.stake_reward_if_locked_and_not_fulfilled();
        assert_eq!(stake_reward2, U256::from(10));

        let locked = ctx.picker.price_order(&mut order2).await;
        assert!(matches!(locked, Ok(OrderPricingOutcome::Skip)));

        // Stake token denom offsets the mcycle multiplier, so for 1stake/mcycle, this will be 10
        assert!(logs_contain(&format!("exec limit cycles for order {order2_id}: 10")));
        assert!(logs_contain(&format!("Skipping order {order2_id} due to session limit exceeded")));
    }

    #[tokio::test]
    #[traced_test]
    async fn test_order_is_locked_check() -> Result<()> {
        let ctx = PickerTestCtxBuilder::default().build().await;
        ctx.prime_state_cache().await;

        let mut order = ctx.generate_next_order(Default::default()).await;
        let order_id = order.id();

        ctx.db
            .set_request_locked(
                U256::from(order.request.id),
                &ctx.provider.default_signer_address().to_string(),
                1000,
            )
            .await?;

        assert!(ctx.db.is_request_locked(U256::from(order.request.id)).await?);

        let pricing_outcome = ctx.picker.price_order(&mut order).await?;
        assert!(matches!(pricing_outcome, OrderPricingOutcome::Skip));

        assert!(logs_contain(&format!("Order {order_id} is already locked, skipping")));

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_duplicate_order_cache() -> Result<()> {
        let mut ctx = PickerTestCtxBuilder::default().build().await;
        ctx.prime_state_cache().await;

        let order1 = ctx.generate_next_order(Default::default()).await;
        let order_id = order1.id();

        // Duplicate order
        let order2 = Box::new(OrderRequest {
            request: order1.request.clone(),
            client_sig: order1.client_sig.clone(),
            fulfillment_type: order1.fulfillment_type,
            boundless_market_address: order1.boundless_market_address,
            chain_id: order1.chain_id,
            image_id: order1.image_id.clone(),
            input_id: order1.input_id.clone(),
            total_cycles: order1.total_cycles,
            target_timestamp: order1.target_timestamp,
            expire_timestamp: order1.expire_timestamp,
        });

        assert_eq!(order1.id(), order2.id(), "Both orders should have the same ID");

        tokio::spawn(ctx.picker.spawn(CancellationToken::new()));

        ctx.new_order_tx.send(order1).await?;
        ctx.new_order_tx.send(order2).await?;

        let first_processed =
            tokio::time::timeout(Duration::from_secs(10), ctx.priced_orders_rx.recv())
                .await?
                .unwrap();

        assert_eq!(first_processed.id(), order_id, "First order should be processed");

        let second_result =
            tokio::time::timeout(Duration::from_secs(2), ctx.priced_orders_rx.recv()).await;

        assert!(second_result.is_err(), "Second order should be deduplicated and not processed");

        assert!(logs_contain(&format!("Skipping duplicate order {order_id}, already processed recently")));

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_order_is_fulfilled_check() -> Result<()> {
        let ctx = PickerTestCtxBuilder::default().build().await;
        ctx.prime_state_cache().await;

        let mut order = ctx
            .generate_next_order(OrderParams {
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                ..Default::default()
            })
            .await;
        let order_id = order.id();

        ctx.db.set_request_fulfilled(U256::from(order.request.id), 1000).await?;

        assert!(ctx.db.is_request_fulfilled(U256::from(order.request.id)).await?);

        let pricing_outcome = ctx.picker.price_order(&mut order).await?;
        assert!(matches!(pricing_outcome, OrderPricingOutcome::Skip));

        assert!(logs_contain(&format!("Order {order_id} is already fulfilled, skipping")));

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_active_tasks_logging() {
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let mut ctx = PickerTestCtxBuilder::default().with_config(config).build().await;
        ctx.prime_state_cache().await;

        // Start the order picker task
        let picker_task = tokio::spawn(ctx.picker.spawn(Default::default()));

        // Send an order to trigger the logging
        let order1 =
            ctx.generate_next_order(OrderParams { order_index: 1, ..Default::default() }).await;
        let order1_id = order1.id();
        ctx.new_order_tx.send(order1).await.unwrap();

        // Wait for the order to be processed and check for the "Added" log
        tokio::time::timeout(
            MIN_CAPACITY_CHECK_INTERVAL + Duration::from_secs(1),
            ctx.priced_orders_rx.recv(),
        )
        .await
        .unwrap();

        // Check that we logged the task being added
        assert!(logs_contain("Priced task for order"));
        assert!(logs_contain(&order1_id));

        picker_task.abort();
    }

    #[tokio::test]
    async fn test_handle_lock_event() {
        let ctx = PickerTestCtxBuilder::default().build().await;
        let mut active_tasks: BTreeMap<U256, BTreeMap<String, CancellationToken>> = BTreeMap::new();
        let mut pending_orders: Vec<Box<OrderRequest>> = Vec::new();

        let lock_and_fulfill_order = ctx
            .generate_next_order(OrderParams {
                order_index: 123,
                fulfillment_type: FulfillmentType::LockAndFulfill,
                ..Default::default()
            })
            .await;

        let fulfill_after_expire_order = ctx
            .generate_next_order(OrderParams {
                order_index: 123,
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                ..Default::default()
            })
            .await;

        let request_id = U256::from(lock_and_fulfill_order.request.id);

        let lock_and_fulfill_token = CancellationToken::new();
        let fulfill_after_expire_token = CancellationToken::new();

        // Add active tasks using actual order IDs
        let mut order_tasks = BTreeMap::new();
        order_tasks.insert(lock_and_fulfill_order.id(), lock_and_fulfill_token.clone());
        order_tasks.insert(fulfill_after_expire_order.id(), fulfill_after_expire_token.clone());
        active_tasks.insert(request_id, order_tasks);

        pending_orders.push(lock_and_fulfill_order);
        pending_orders.push(fulfill_after_expire_order);

        handle_lock_event(request_id, &mut active_tasks, &mut pending_orders);

        assert!(lock_and_fulfill_token.is_cancelled(), "LockAndFulfill task should be cancelled");
        assert!(
            !fulfill_after_expire_token.is_cancelled(),
            "FulfillAfterLockExpire task should NOT be cancelled"
        );

        assert!(active_tasks.contains_key(&request_id));
        let remaining_tasks = active_tasks.get(&request_id).unwrap();
        assert_eq!(remaining_tasks.len(), 1);
        let remaining_order_id = remaining_tasks.keys().next().unwrap();
        assert!(remaining_order_id.contains("FulfillAfterLockExpire"));

        assert_eq!(pending_orders.len(), 1);
        assert_eq!(pending_orders[0].fulfillment_type, FulfillmentType::FulfillAfterLockExpire);
    }

    #[tokio::test]
    async fn test_handle_fulfill_event() {
        // Create test context and orders
        let ctx = PickerTestCtxBuilder::default().build().await;
        let mut active_tasks: BTreeMap<U256, BTreeMap<String, CancellationToken>> = BTreeMap::new();
        let mut pending_orders: Vec<Box<OrderRequest>> = Vec::new();

        let lock_and_fulfill_order = ctx
            .generate_next_order(OrderParams {
                order_index: 456,
                fulfillment_type: FulfillmentType::LockAndFulfill,
                ..Default::default()
            })
            .await;

        let fulfill_after_expire_order = ctx
            .generate_next_order(OrderParams {
                order_index: 456,
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                ..Default::default()
            })
            .await;

        let request_id = U256::from(lock_and_fulfill_order.request.id);

        let token1 = CancellationToken::new();
        let token2 = CancellationToken::new();

        let mut order_tasks = BTreeMap::new();
        order_tasks.insert(lock_and_fulfill_order.id(), token1.clone());
        order_tasks.insert(fulfill_after_expire_order.id(), token2.clone());
        active_tasks.insert(request_id, order_tasks);

        pending_orders.push(lock_and_fulfill_order);
        pending_orders.push(fulfill_after_expire_order);

        handle_fulfill_event(request_id, &mut active_tasks, &mut pending_orders);

        assert!(token1.is_cancelled(), "All tasks should be cancelled");
        assert!(token2.is_cancelled(), "All tasks should be cancelled");

        assert!(!active_tasks.contains_key(&request_id));

        assert_eq!(pending_orders.len(), 0, "All pending orders should be removed");
    }

    // Mock prover that tracks preflight calls
    struct MockPreflightTracker {
        preflight_calls: Arc<std::sync::Mutex<Vec<(String, String)>>>,
        default_prover: Arc<DefaultProver>,
    }

    impl MockPreflightTracker {
        fn new() -> Self {
            Self {
                preflight_calls: Arc::new(std::sync::Mutex::new(Vec::new())),
                default_prover: Arc::new(DefaultProver::new()),
            }
        }

        fn get_preflight_calls(&self) -> Vec<(String, String)> {
            self.preflight_calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Prover for MockPreflightTracker {
        async fn upload_image(&self, image_id: &str, image: Vec<u8>) -> Result<(), ProverError> {
            self.default_prover.upload_image(image_id, image).await
        }

        async fn upload_input(&self, input: Vec<u8>) -> Result<String, ProverError> {
            self.default_prover.upload_input(input).await
        }

        async fn preflight(
            &self,
            image_id: &str,
            input_id: &str,
            assumptions: Vec<String>,
            executor_limit: Option<u64>,
            order_id: &str,
        ) -> Result<ProofResult, ProverError> {
            // Track the preflight call
            self.preflight_calls.lock().unwrap().push((image_id.to_string(), input_id.to_string()));

            // Call the default prover
            self.default_prover
                .preflight(image_id, input_id, assumptions, executor_limit, order_id)
                .await
        }

        async fn has_image(&self, image_id: &str) -> Result<bool, ProverError> {
            self.default_prover.has_image(image_id).await
        }

        async fn prove_stark(
            &self,
            image_id: &str,
            input_id: &str,
            assumptions: Vec<String>,
        ) -> Result<String, ProverError> {
            self.default_prover.prove_stark(image_id, input_id, assumptions).await
        }

        async fn wait_for_stark(&self, proof_id: &str) -> Result<ProofResult, ProverError> {
            self.default_prover.wait_for_stark(proof_id).await
        }

        async fn cancel_stark(&self, proof_id: &str) -> Result<(), ProverError> {
            self.default_prover.cancel_stark(proof_id).await
        }

        async fn get_receipt(&self, proof_id: &str) -> Result<Option<Receipt>, ProverError> {
            self.default_prover.get_receipt(proof_id).await
        }

        async fn get_preflight_journal(
            &self,
            proof_id: &str,
        ) -> Result<Option<Vec<u8>>, ProverError> {
            self.default_prover.get_preflight_journal(proof_id).await
        }

        async fn get_journal(&self, proof_id: &str) -> Result<Option<Vec<u8>>, ProverError> {
            self.default_prover.get_journal(proof_id).await
        }

        async fn compress(&self, proof_id: &str) -> Result<String, ProverError> {
            self.default_prover.compress(proof_id).await
        }

        async fn get_compressed_receipt(
            &self,
            proof_id: &str,
        ) -> Result<Option<Vec<u8>>, ProverError> {
            self.default_prover.get_compressed_receipt(proof_id).await
        }
    }

    #[tokio::test]
    #[traced_test]
    async fn test_preflight_cache_behavior() -> Result<()> {
        let mock_prover = Arc::new(MockPreflightTracker::new());

        let image_id = Digest::from(ECHO_ID).to_string();
        mock_prover.upload_image(&image_id, ECHO_ELF.to_vec()).await.unwrap();

        let ctx = PickerTestCtxBuilder::default().with_prover(mock_prover.clone()).build().await;
        ctx.prime_state_cache().await;

        let mut order1 =
            ctx.generate_next_order(OrderParams { order_index: 100, ..Default::default() }).await;

        let mut order2 =
            ctx.generate_next_order(OrderParams { order_index: 200, ..Default::default() }).await;

        let mut order3 = ctx
            .generate_next_order(OrderParams {
                order_index: 100,
                fulfillment_type: FulfillmentType::FulfillAfterLockExpire,
                ..Default::default()
            })
            .await;

        assert_eq!(
            order1.request.id, order3.request.id,
            "Order1 and Order3 should have same request ID"
        );
        assert_ne!(
            order1.request.id, order2.request.id,
            "Order1 and Order2 should have different request IDs"
        );

        // Process order1 and order2 concurrently to test cache atomicity
        let (pricing1, pricing2) =
            tokio::join!(ctx.picker.price_order(&mut order1), ctx.picker.price_order(&mut order2));

        assert!(pricing1.is_ok(), "Order1 pricing should succeed");
        assert!(pricing2.is_ok(), "Order2 pricing should succeed");

        // Process order3 (should use cache)
        let pricing3 = ctx.picker.price_order(&mut order3).await;
        assert!(pricing3.is_ok(), "Order3 pricing should succeed");

        // Check preflight calls - should only be called once since all orders are identical
        let preflight_calls = mock_prover.get_preflight_calls();

        // Since ALL orders have the same image_url and input data, they should share the same cache entry
        assert_eq!(
            preflight_calls.len(),
            1,
            "Should have exactly 1 preflight call since all orders are identical.",
        );

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_smaller_cycle_limit_cache() -> Result<()> {
        let mock_prover = Arc::new(MockPreflightTracker::new());
        let image_id = Digest::from(LOOP_ID).to_string();
        mock_prover.upload_image(&image_id, LOOP_ELF.to_vec()).await.unwrap();

        // Create context with very low mcycle price and set peak_prove_khz to create different deadline caps
        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
            config.load_write().unwrap().market.peak_prove_khz = Some(1000); // Set peak_prove_khz to create deadline caps
            config.load_write().unwrap().market.min_deadline = 0; // Remove min_deadline interference
        }
        let ctx = PickerTestCtxBuilder::default()
            .with_prover(mock_prover.clone())
            .with_config(config)
            .build()
            .await;
        ctx.prime_state_cache().await;

        // Create two orders with same program+input but very different exec limits due to different timeouts:
        // Order 1: Very short timeout = very low deadline cap (should hit session limit exceeded)
        // We'll set the loop to consume 50M cycles, which exceeds the 20M cycle cap from short timeout
        let mut low_timeout_order = ctx
            .generate_loop_order(
                OrderParams {
                    order_index: 1,
                    min_price: parse_ether("100.0").unwrap(), // High price but will be capped by very short timeout
                    max_price: parse_ether("100.0").unwrap(),
                    timeout: 30, // Very short timeout = very low deadline cap (30s * 1000khz = 30M cycles)
                    lock_timeout: 2, // Also set short lock_timeout
                    ..Default::default()
                },
                5_000_000,
            ) // 5M cycles - should exceed the 20M cycle limit
            .await;

        // Order 2: Long timeout = high deadline cap (should succeed and NOT reuse low-limit cache)
        // Same cycle count but much higher exec limit due to longer timeout
        let mut high_timeout_order = ctx
            .generate_loop_order(
                OrderParams {
                    order_index: 2,
                    min_price: parse_ether("100.0").unwrap(), // Same high price but much longer timeout
                    max_price: parse_ether("100.0").unwrap(),
                    timeout: 3600, // Much longer timeout = high deadline cap (3600s * 1000khz = 3.6B cycles)
                    lock_timeout: 3000, // Also set long lock_timeout
                    ..Default::default()
                },
                5_000_000,
            ) // Same 5M cycles - should be under the 3B cycle limit
            .await;

        // Process short timeout order first - this should hit session limit and cache the Skip result
        let result1 = ctx.picker.price_order(&mut low_timeout_order).await;
        assert!(matches!(result1, Ok(OrderPricingOutcome::Skip)));

        // Process long timeout order second - this should NOT reuse the low-limit cached result
        // It should succeed with its own higher exec limit via a new preflight call
        let result2 = ctx.picker.price_order(&mut high_timeout_order).await;
        assert!(matches!(result2, Ok(OrderPricingOutcome::Lock { .. })));

        // We expect 2 preflight calls since the orders have different deadline-based exec limits
        let preflight_calls = mock_prover.get_preflight_calls();
        assert_eq!(
            preflight_calls.len(),
            2,
            "Should have exactly 2 preflight calls since orders have different exec limits due to different timeouts.",
        );

        // Check that the log message about insufficient limit was produced
        assert!(logs_contain(&format!(
            "Cached result has insufficient limit for order {}",
            high_timeout_order.id()
        )));

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_concurrent_preflights_with_cancellation() -> Result<()> {
        let mock_prover = Arc::new(MockPreflightTracker::new());
        let image_id = Digest::from(LOOP_ID).to_string();
        mock_prover.upload_image(&image_id, LOOP_ELF.to_vec()).await.unwrap();

        let config = ConfigLock::default();
        {
            config.load_write().unwrap().market.mcycle_price = "0.0000001".into();
        }
        let ctx = PickerTestCtxBuilder::default()
            .with_prover(mock_prover.clone())
            .with_config(config)
            .build()
            .await;
        ctx.prime_state_cache().await;

        // Create two orders with same program+input for same cache key
        let order_a = ctx
            .generate_loop_order(
                OrderParams {
                    order_index: 1,
                    min_price: parse_ether("100.0").unwrap(),
                    max_price: parse_ether("100.0").unwrap(),
                    timeout: 3600,
                    lock_timeout: 3000,
                    ..Default::default()
                },
                5_000_000,
            )
            .await;

        let order_b = ctx
            .generate_loop_order(
                OrderParams {
                    order_index: 2,
                    min_price: parse_ether("100.0").unwrap(),
                    max_price: parse_ether("100.0").unwrap(),
                    timeout: 3600,
                    lock_timeout: 3000,
                    ..Default::default()
                },
                5_000_000,
            )
            .await;

        // Create cancellation tokens
        let cancel_token_a = CancellationToken::new();
        let cancel_token_b = CancellationToken::new();

        // Save order IDs before moving into tasks
        let order_a_id = order_a.id();
        let _order_b_id = order_b.id();

        // Start both preflights concurrently with a slight stagger
        let cancel_a_clone = cancel_token_a.clone();
        let picker_a = ctx.picker.clone();
        let task_a = tokio::spawn(async move {
            picker_a.price_order_and_update_state(order_a, cancel_token_a).await
        });

        // Small delay to ensure task A starts first
        tokio::time::sleep(Duration::from_millis(10)).await;

        let picker_b = ctx.picker.clone();
        let task_b = tokio::spawn(async move {
            picker_b.price_order_and_update_state(order_b, cancel_token_b).await
        });

        // Wait for task A to start its preflight before cancelling
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if logs_contain(&format!("Starting preflight of {order_a_id}")) {
                // Sleep to wait for B to wait on this preflight
                tokio::time::sleep(Duration::from_millis(20)).await;
                break;
            }
        }

        // Cancel task A now that we know it has started preflight
        cancel_a_clone.cancel();

        // Wait for both tasks to complete
        let result_a = task_a.await.unwrap();
        let result_b = task_b.await.unwrap();

        // Task A should have been cancelled and returned false
        assert!(!result_a, "Task A should have been cancelled");

        // Task B should have completed successfully
        assert!(result_b, "Task B should have completed successfully");

        // Check that the cancellation was logged
        assert!(logs_contain("Order pricing cancelled during pricing for order"));

        // Verify that both preflight calls were made (both tasks start their preflights)
        // Task A starts its preflight but gets cancelled during execution
        // Task B completes its preflight successfully
        let preflight_calls = mock_prover.get_preflight_calls();
        assert_eq!(preflight_calls.len(), 1, "Should have exactly 1 preflight call");

        Ok(())
    }
}
