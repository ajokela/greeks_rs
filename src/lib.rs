use pyo3::prelude::*;
use std::f64::consts::E;
use pyo3::exceptions::PyValueError;
use statrs::distribution::Normal;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Continuous;

/// Calculate the price of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The price of the European option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let price = black_scholes_price(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Option price: {}", price);
/// ```
#[pyfunction]
fn black_scholes_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q);
    let d2 = d1 - sigma * f64::sqrt(t);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let price = match option_type {
        "call" => s * f64::exp(-q * t) * norm.cdf(d1) - k * f64::exp(-r * t) * norm.cdf(d2),
        "put" => k * f64::exp(-r * t) * norm.cdf(-d2) - s * f64::exp(-q * t) * norm.cdf(-d1),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(price)
}

/// Calculate the delta (sensitivity to the underlying price) of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options. The delta
/// measures the rate of change of the option price with respect to changes in the price of the underlying
/// asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///                  If not provided, it defaults to 'call'.
///
/// # Returns
///
/// The delta of the European option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let delta = black_scholes_delta(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, Some("call")).unwrap();
/// println!("Delta: {}", delta);
/// ```
#[pyfunction]
fn black_scholes_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: Option<&str>) -> PyResult<f64> {
    let option_type = option_type.unwrap_or("call");
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();

    let delta = match option_type {
        "call" => f64::exp(-q * t) * norm.cdf(d1),
        "put" => -f64::exp(-q * t) * norm.cdf(-d1),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    Ok(delta)
}

/// Calculate the gamma (second-order sensitivity to the underlying price) of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options. The gamma
/// measures the rate of change of the option's delta with respect to changes in the price of the
/// underlying asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
///
/// # Returns
///
/// The gamma of the European option.
///
/// # Example
///
/// ```rust
/// let gamma = black_scholes_gamma(100.0, 110.0, 1.0, 0.05, 0.2, 0.0).unwrap();
/// println!("Gamma: {}", gamma);
/// ```
#[pyfunction]
fn black_scholes_gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) ->  PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm_dist = Normal::new(0.0, 1.0).unwrap();
    let gamma = norm_dist.pdf(d1) * E.powf(-q * t) / (s * sigma * f64::sqrt(t));

    Ok(gamma)
}

/// Calculate the theta (sensitivity to time to expiration) of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options. The theta
/// measures the sensitivity of the option price to changes in the time to expiration, expressed as the
/// change in option price per day.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///                  If not provided, it defaults to 'call'.
///
/// # Returns
///
/// The theta of the European option, expressed as the change in option price per day.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let theta = black_scholes_theta(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, Some("call")).unwrap();
/// println!("Theta: {}", theta);
/// ```
#[pyfunction]
fn black_scholes_theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: Option<&str>) -> PyResult<f64> {
    let option_type = option_type.unwrap_or("call");
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let d2 = d1 - sigma * f64::sqrt(t);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let theta = match option_type {
        "call" => {
            - (s * sigma * f64::exp(-q * t) * norm.pdf(d1)) / (2.0 * f64::sqrt(t))
            - r * k * f64::exp(-r * t) * norm.cdf(d2)
            + q * s * f64::exp(-q * t) * norm.cdf(d1)
        },
        "put" => {
            - (s * sigma * f64::exp(-q * t) * norm.pdf(d1)) / (2.0 * f64::sqrt(t))
            + r * k * f64::exp(-r * t) * norm.cdf(-d2)
            - q * s * f64::exp(-q * t) * norm.cdf(-d1)
        },
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    // Convert to per-day theta
    Ok(theta / 365.0)
}

/// Calculate the vega (sensitivity to volatility) of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options. The vega
/// measures the sensitivity of the option price to changes in the volatility of the underlying asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
///
/// # Returns
///
/// The vega of the European option.
///
/// # Example
///
/// ```rust
/// let vega = black_scholes_vega(100.0, 110.0, 1.0, 0.05, 0.2, 0.0).unwrap();
/// println!("Vega: {}", vega);
/// ```
#[pyfunction]
fn black_scholes_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();

    let vega = s * f64::exp(-q * t) * norm.pdf(d1) * f64::sqrt(t);

    Ok(vega)
}

/// Calculate the rho (sensitivity to interest rate) of a European option using the Black-Scholes model.
///
/// The Black-Scholes model is a widely used mathematical model for pricing European options. The rho
/// measures the sensitivity of the option price to changes in the risk-free interest rate.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///                  If not provided, it defaults to 'call'.
///
/// # Returns
///
/// The rho of the European option, divided by 100 to represent the change in option price for a 1%
/// change in the risk-free interest rate.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let rho = black_scholes_rho(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, Some("call")).unwrap();
/// println!("Rho: {}", rho);
/// ```
#[pyfunction]
fn black_scholes_rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: Option<&str>) -> PyResult<f64> {
    let option_type = option_type.unwrap_or("call");
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64; 
    let d2 = d1 - sigma * f64::sqrt(t);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let rho = match option_type {
        "call" => k * t * f64::exp(-r * t) * norm.cdf(d2),
        "put" => -k * t * f64::exp(-r * t) * norm.cdf(-d2),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    Ok(rho / 100.0)
}

/// Calculate the price of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The price of the American option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let price = bjerksund_stensland_price(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Option price: {}", price);
/// ```
#[pyfunction]
fn bjerksund_stensland_price(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;
    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

    let (d1, d2, d3, d4) = (
        (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
        (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
        ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
        ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
    );

    let n = Normal::new(0.0, 1.0).unwrap();
    let (n_d1, n_d2, n_d3, n_d4) = (n.cdf(d1), n.cdf(d2), n.cdf(d3), n.cdf(d4));
    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };
    let kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t)))
            * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };
    let price = match option_type {
        "call" => {
            s * E.powf(-q * t) * n_d1 - k * E.powf(-r * t) * n_d2
                + s * E.powf(-q * t) * (1.0 - E.powf(-(r - b) * t)) * (s / b_t_infinity).powf(x) * (n_d2 - kappa)
        }
        "put" => {
            k * E.powf(-r * t) * n_d4 - s * E.powf(-q * t) * n_d3
                + s * E.powf(-q * t) * (1.0 - E.powf(-(r - b) * t)) * (s / b_t_zero).powf(-x) * (n_d4 + kappa)
        }
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };
    Ok(price)
}

/// Calculate the delta (sensitivity to the underlying price) of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise. The delta measures the rate of change of the option price with respect to changes
/// in the price of the underlying asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The delta of the American option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let delta = bjerksund_stensland_delta(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Delta: {}", delta);
/// ```
#[pyfunction]
fn bjerksund_stensland_delta(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;
    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

    let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
    let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

    let n = Normal::new(0.0, 1.0).unwrap();
    let n_d1 = n.cdf(d1);
    let n_d2 = n.cdf(d2);
    let n_d3 = n.cdf(d3);
    let n_d4 = n.cdf(d4);

    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };

    let delta = match option_type {
        "call" => n_d1 - (s / b_t_infinity).powf(x) * (n_d2 - kappa),
        "put" => -n_d3 + (s / b_t_zero).powf(-x) * (n_d4 + kappa),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(delta)
}

/// Calculate the gamma (second-order sensitivity to the underlying price) of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise. The gamma measures the rate of change of the option's delta with respect to changes
/// in the price of the underlying asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The gamma of the American option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let gamma = bjerksund_stensland_gamma(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Gamma: {}", gamma);
/// ```
#[pyfunction]
fn bjerksund_stensland_gamma(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;

    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

    let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
    let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

    let n = Normal::new(0.0, 1.0).unwrap();
    let _n_d1 = n.cdf(d1);
    let n_d2 = n.cdf(d2);
    let _n_d3 = n.cdf(d3);
    let n_d4 = n.cdf(d4);
    let n_prime_d1 = n.pdf(d1);
    let n_prime_d2 = n.pdf(d2);
    let n_prime_d3 = n.pdf(d3);
    let n_prime_d4 = n.pdf(d4);

    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };

    let kappa_prime = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.pdf(y) * (-alpha / (sigma * t_sqrt)) - (s / b_t_infinity).powf(beta) * n.pdf(y - 2.0 * alpha / (sigma * t_sqrt)) * (-alpha / (sigma * t_sqrt)) * (beta - 1.0))
    };

    let gamma = match option_type {
        "call" => {
            n_prime_d1 / (s * sigma * t_sqrt) - x * (s / b_t_infinity).powf(x - 1.0) * (n_d2 - kappa) / b_t_infinity - (s / b_t_infinity).powf(x) * (n_prime_d2 * d2 / (sigma * t_sqrt) - kappa_prime) / s
        },
        "put" => {
            n_prime_d3 / (s * sigma * t_sqrt) + x * (s / b_t_zero).powf(-x - 1.0) * (n_d4 + kappa) / b_t_zero + (s / b_t_zero).powf(-x) * (-n_prime_d4 * d4 / (sigma * t_sqrt) + kappa_prime) / s
        },
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(gamma)
}

/// Calculate the theta (sensitivity to time to expiration) of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise. The theta measures the sensitivity of the option price to changes in the time to
/// expiration, expressed as the change in option price per day.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The theta of the American option, expressed as the change in option price per day.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let theta = bjerksund_stensland_theta(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Theta: {}", theta);
/// ```
#[pyfunction]
fn bjerksund_stensland_theta(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;
    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

    let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
    let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

    let n = Normal::new(0.0, 1.0).unwrap();
    let _n_d1 = n.cdf(d1);
    let n_d2 = n.cdf(d2);
    let _n_d3 = n.cdf(d3);
    let n_d4 = n.cdf(d4);
    let n_prime_d1 = n.pdf(d1);
    let _n_prime_d2 = n.pdf(d2);
    let n_prime_d3 = n.pdf(d3);
    let _n_prime_d4 = n.pdf(d4);

    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };

    let _kappa_prime = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.pdf(y) * (-alpha / (sigma * t_sqrt)) - (s / b_t_infinity).powf(beta) * n.pdf(y - 2.0 * alpha / (sigma * t_sqrt)) * (-alpha / (sigma * t_sqrt)) * (beta - 1.0))
    };

    let _kappa_double_prime = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.pdf(y) * (alpha.powi(2) / (sigma.powi(2) * t) - y / (sigma * t_sqrt)) - (s / b_t_infinity).powf(beta) * n.pdf(y - 2.0 * alpha / (sigma * t_sqrt)) * ((2.0 * alpha.powi(2) * (beta - 1.0)) / (sigma.powi(2) * t) - (y - 2.0 * alpha / (sigma * t_sqrt)) * (alpha * (beta - 1.0)) / (sigma * t_sqrt)))
    };

    let theta = match option_type {
        "call" => {
            -s * n_prime_d1 * sigma / (2.0 * t_sqrt) - r * k * E.powf(-r * t) * n_d2 + r * b_t_infinity * (s / b_t_infinity).powf(x) * (n_d2 - kappa)
        },
        "put" => {
            -s * n_prime_d3 * sigma / (2.0 * t_sqrt) + r * k * E.powf(-r * t) * n_d4 - r * b_t_zero * (s / b_t_zero).powf(-x) * (n_d4 + kappa)
        },
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(theta / 365.0)
}

/// Calculate the vega (sensitivity to volatility) of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise. The vega measures the sensitivity of the option price to changes in the volatility
/// of the underlying asset.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The vega of the American option.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let vega = bjerksund_stensland_vega(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Vega: {}", vega);
/// ```
#[pyfunction]
fn bjerksund_stensland_vega(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;
    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };
    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let _x = 2.0 * r / (sigma.powi(2) * (1.0 - std::f64::consts::E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - std::f64::consts::E.powf(-b * t) + epsilon));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * std::f64::consts::E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * std::f64::consts::E.powf(h_t);

    let (d1, d2, d3, d4) = (
        (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
        (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt),
        ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
        ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt),
    );

    let n = Normal::new(0.0, 1.0).unwrap();
    let n_prime_d1 = if d1.abs() > 10.0 { 1e-10 } else { n.pdf(d1) };
    let _n_prime_d2 = n.pdf(d2);
    let n_prime_d3 = if d3.abs() > 10.0 { 1e-10 } else { n.pdf(d3) };
    let _n_prime_d4 = n.pdf(d4);
    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };
    let _kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - std::f64::consts::E.powf(-r * t)))
            * (std::f64::consts::E.powf(alpha) * n.cdf(y)
                - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };
    let vega = match option_type {
        "call" => {
            s * t_sqrt * n_prime_d1 * (1.0 - (s / b_t_infinity).powf(beta) * n.cdf(y))
        }
        "put" => {
            s * t_sqrt * n_prime_d3 * (1.0 - (s / b_t_zero).powf(-beta) * n.cdf(-y))
        }
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };
    Ok(vega)
}

/// Calculate the rho (sensitivity to interest rate) of an American option using the Bjerksund-Stensland model.
///
/// The Bjerksund-Stensland model is used to price American options, taking into account the possibility
/// of early exercise. The rho measures the sensitivity of the option price to changes in the risk-free
/// interest rate.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
/// * `option_type` - The type of option. Use 'call' for a call option and 'put' for a put option.
///
/// # Returns
///
/// The rho of the American option. The rho is divided by 100 to represent the change in option price
/// for a 1% change in the risk-free interest rate.
///
/// # Errors
///
/// Returns a `PyValueError` if an invalid option type is provided. Valid option types are 'call' and 'put'.
///
/// # Example
///
/// ```rust
/// let rho = bjerksund_stensland_rho(100.0, 110.0, 1.0, 0.05, 0.2, 0.0, "call").unwrap();
/// println!("Rho: {}", rho);
/// ```
#[pyfunction]
fn bjerksund_stensland_rho(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    q: f64,
    option_type: &str,
) -> PyResult<f64> {
    let epsilon = 0.00001;
    let t_sqrt = t.sqrt();
    let b = q;
    let beta = (0.5 - b / sigma.powi(2)) + ((b / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_infinity = beta / (beta - 1.0) * k;
    let b_zero = match option_type {
        "call" => k.max(r / (r - b) * k),
        "put" => k.min(r / (r - b) * k),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let h_t = -(b * t + 2.0 * sigma * t_sqrt) * (k / (b_infinity - b_zero));
    let x = 2.0 * r / (sigma.powi(2) * (1.0 - E.powf(-r * t)));
    let y = 2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-b * t)));
    let b_t_infinity = b_infinity - (b_infinity - b_zero) * E.powf(h_t);
    let b_t_zero = b_zero + (b_infinity - b_zero) * E.powf(h_t);

    let d1 = (s / b_t_infinity).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d2 = (b_t_infinity / s).ln() + (b + sigma.powi(2) / 2.0) * t / (sigma * t_sqrt);
    let d3 = ((s / b_t_zero).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);
    let d4 = ((b_t_zero / s).ln() + (b + sigma.powi(2) / 2.0) * t) / (sigma * t_sqrt);

    let n = Normal::new(0.0, 1.0).unwrap();
    let _n_d1 = n.cdf(d1);
    let n_d2 = n.cdf(d2);
    let _n_d3 = n.cdf(d3);
    let n_d4 = n.cdf(d4);

    let (alpha, beta) = match option_type {
        "call" => (-(r - b) * t - 2.0 * sigma * t_sqrt, 2.0 * (r - b) / sigma.powi(2)),
        "put" => ((r - b) * t + 2.0 * sigma * t_sqrt, -2.0 * (r - b) / sigma.powi(2)),
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    let kappa = if b >= r || b <= epsilon {
        0.0
    } else {
        2.0 * b / (sigma.powi(2) * (1.0 - E.powf(-r * t))) * (E.powf(alpha) * n.cdf(y) - (s / b_t_infinity).powf(beta) * n.cdf(y - 2.0 * alpha / (sigma * t_sqrt)))
    };

    let rho = match option_type {
        "call" => {
            k * t * E.powf(-r * t) * n_d2 + t * b_t_infinity * (s / b_t_infinity).powf(x) * (n_d2 - kappa)
        },
        "put" => {
            -k * t * E.powf(-r * t) * n_d4 - t * b_t_zero * (s / b_t_zero).powf(-x) * (n_d4 + kappa)
        },
        _ => return Err(PyErr::new::<PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(rho / 100.0)
}

/// Calculate the d1 parameter for the Black-Scholes model.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset.
/// * `k` - The strike price of the option.
/// * `t` - The time to expiration of the option in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility of the underlying asset.
/// * `q` - The dividend yield of the underlying asset.
///
/// # Returns
///
/// The d1 parameter.
fn calculate_d1(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> f64 {
    (f64::ln(s / k) + (r - q + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t))
}

#[pymodule]
fn libgai_stocks(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(black_scholes_delta, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_theta, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_vega, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_rho, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;

    m.add_function(wrap_pyfunction!(bjerksund_stensland_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_theta, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_vega, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_rho, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_price, m)?)?;

    Ok(())
}