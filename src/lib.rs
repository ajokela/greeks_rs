use pyo3::prelude::*;
use statrs::distribution::{Normal, ContinuousCDF, Continuous};
use std::f64::consts::E;

/**
 * Calculate the delta of an approximate American option using the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The delta of the option.
 */
#[pyfunction]
fn black_scholes_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();

    let delta = match option_type {
        "call" => f64::exp(-q * t) * norm.cdf(d1),
        "put" => -f64::exp(-q * t) * norm.cdf(-d1),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    Ok(delta)
}

/**
 * Calculate the gamma of an approximate American option using the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * 
 * Returns
 * -------
 * float
 *     The gamma of the option.
 */
#[pyfunction]
fn black_scholes_gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> f64 {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm_dist = Normal::new(0.0, 1.0).unwrap();
    let gamma = norm_dist.pdf(d1) * E.powf(-q * t) / (s * sigma * f64::sqrt(t));

    gamma
}

/**
 * Calculate the theta of an approximate American option using the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The theta of the option.
 */

#[pyfunction]
fn black_scholes_theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
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
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    // Convert to per-day theta
    Ok(theta / 365.0)
}

/**
 * Calculate the vega of an approximate American option using the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * 
 * Returns
 * -------
 * float
 *     The vega of the option.
 */
#[pyfunction]
fn black_scholes_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64;
    let norm = Normal::new(0.0, 1.0).unwrap();

    let vega = s * f64::exp(-q * t) * norm.pdf(d1) * f64::sqrt(t);

    Ok(vega)
}

/**
 * Calculate the rho of an approximate American option using the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The rho of the option.
 */
#[pyfunction]
fn black_scholes_rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let d1 = calculate_d1(s, k, t, r, sigma, q) as f64; 
    let d2 = d1 - sigma * f64::sqrt(t);
    let norm = Normal::new(0.0, 1.0).unwrap();

    let rho = match option_type {
        "call" => k * t * f64::exp(-r * t) * norm.cdf(d2),
        "put" => -k * t * f64::exp(-r * t) * norm.cdf(-d2),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'."))
    };

    Ok(rho)
}

/**
 * Calculate the d1 parameter for the Black-Scholes model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * 
 * Returns
 * -------
 * float
 *     The d1 parameter.
 */
fn calculate_d1(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> f64 {
    (f64::ln(s / k) + (r - q + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t))
}

// #################################################################################
// 
// Bjerksund-Stensland model

/**
 * Calculate the delta of an approximate American option using the Bjerksund-Stensland model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The delta of the option.
 */
#[pyfunction]
fn bjerksund_stensland_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let beta = (0.5 - q / sigma.powi(2)) + ((q / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_max = k.max(r / (r - q) * k);

    let b_infinity = beta / (beta - 1.0) * k;
    let b = b_max.min(b_infinity);

    let x1 = b + (b - k) * (1.0 - f64::exp((beta - 1.0) * (f64::ln(b) - f64::ln(k))));

    /*
     * 'y' could be used for calculating the price of the option 
     */
     let _y = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + (beta - 0.5) * sigma * t.sqrt();

    let b1 = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + beta * sigma * t.sqrt();

    let norm = Normal::new(0.0, 1.0).unwrap();
    let delta = match option_type {
        "call" => {
            let delta_call = f64::exp(-q * t) * norm.cdf(b1);
            delta_call
        },
        "put" => {
            let delta_put = f64::exp(-q * t) * (norm.cdf(b1) - 1.0);
            delta_put
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(delta)
}

/**
 * Calculate the gamma of an approximate American option using the Bjerksund-Stensland model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * 
 * Returns
 * -------
 * float
 *     The gamma of the option.
 */
#[pyfunction]
fn bjerksund_stensland_gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
    let beta = (0.5 - q / sigma.powi(2)) + ((q / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_max = k.max(r / (r - q) * k);

    let b_infinity = beta / (beta - 1.0) * k;
    let b = b_max.min(b_infinity);

    let x1 = b + (b - k) * (1.0 - f64::exp((beta - 1.0) * (f64::ln(b) - f64::ln(k))));
    
    /*
     * 'y' could be used for calculating the price of the option 
     */
    let _y = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + (beta - 0.5) * sigma * t.sqrt();

    let b1 = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + beta * sigma * t.sqrt();

    let norm = Normal::new(0.0, 1.0).unwrap();
    let gamma = f64::exp(-q * t) * norm.pdf(b1) / (s * sigma * t.sqrt());

    Ok(gamma)
}

/**
 * Calculate the theta of an approximate American option using the Bjerksund-Stensland model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The theta of the option.
 */
#[pyfunction]
fn bjerksund_stensland_theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let beta = (0.5 - q / sigma.powi(2)) + ((q / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_max = k.max(r / (r - q) * k);

    let b_infinity = beta / (beta - 1.0) * k;
    let b = b_max.min(b_infinity);

    let x1 = b + (b - k) * (1.0 - f64::exp((beta - 1.0) * (f64::ln(b) - f64::ln(k))));

    /*
     * 'y' could be used for calculating the price of the option 
     */
    let _y = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + (beta - 0.5) * sigma * t.sqrt();

    let b1 = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + beta * sigma * t.sqrt();

    let norm = Normal::new(0.0, 1.0).unwrap();
    let theta = match option_type {
        "call" => {
            let theta_call = -s * f64::exp(-q * t) * norm.pdf(b1) * sigma / (2.0 * t.sqrt())
                - r * x1 * f64::exp(-r * t) * norm.cdf(b1 - sigma * t.sqrt())
                + q * s * f64::exp(-q * t) * norm.cdf(b1);
            theta_call / 365.0
        },
        "put" => {
            let theta_put = -s * f64::exp(-q * t) * norm.pdf(b1) * sigma / (2.0 * t.sqrt())
                + r * x1 * f64::exp(-r * t) * norm.cdf(-b1 + sigma * t.sqrt())
                - q * s * f64::exp(-q * t) * norm.cdf(-b1);
            theta_put / 365.0
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(theta)
}

/**
 * Calculate the vega of an approximate American option using the Bjerksund-Stensland model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * 
 * Returns
 * -------
 * float
 *     The vega of the option.
 */
#[pyfunction]
fn bjerksund_stensland_vega(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
    let beta = (0.5 - q / sigma.powi(2)) + ((q / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_max = k.max(r / (r - q) * k);

    let b_infinity = beta / (beta - 1.0) * k;
    let b = b_max.min(b_infinity);

    let x1 = b + (b - k) * (1.0 - f64::exp((beta - 1.0) * (f64::ln(b) - f64::ln(k))));
    
    /*
     * 'y' could be used for calculating the price of the option 
     */
    let _y = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + (beta - 0.5) * sigma * t.sqrt();

    let b1 = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + beta * sigma * t.sqrt();
 
    let norm = Normal::new(0.0, 1.0).unwrap();
    let vega = s * f64::exp(-q * t) * norm.pdf(b1) * t.sqrt();

    Ok(vega)
}

/**
 * Calculate the rho of an approximate American option using the Bjerksund-Stensland model.
 * 
 * Parameters
 * ----------
 * s : float
 *     The current price of the underlying asset.
 * k : float
 *     The strike price of the option.
 * t : float
 *     The time to expiration of the option in years.
 * r : float
 *     The risk-free interest rate.
 * sigma : float
 *     The volatility of the underlying asset.
 * q : float
 *     The dividend yield of the underlying asset.
 * option_type : str
 *     The type of option. Use 'call' for a call option and 'put' for a put option.
 * 
 * Returns
 * -------
 * float
 *     The rho of the option.
 */
#[pyfunction]
fn bjerksund_stensland_rho(s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, option_type: &str) -> PyResult<f64> {
    let beta = (0.5 - q / sigma.powi(2)) + ((q / sigma.powi(2) - 0.5).powi(2) + 2.0 * r / sigma.powi(2)).sqrt();
    let b_max = k.max(r / (r - q) * k);

    let b_infinity = beta / (beta - 1.0) * k;
    let b = b_max.min(b_infinity);

    let x1 = b + (b - k) * (1.0 - f64::exp((beta - 1.0) * (f64::ln(b) - f64::ln(k))));
    
    /*
     * 'y' could be used for calculating the price of the option 
     */
    let _y = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + (beta - 0.5) * sigma * t.sqrt();

    let b1 = (f64::ln(s) - f64::ln(x1)) / (sigma * t.sqrt()) + beta * sigma * t.sqrt();
 
    let norm = Normal::new(0.0, 1.0).unwrap();
    let rho = match option_type {
        "call" => {
            let rho_call = t * x1 * f64::exp(-r * t) * norm.cdf(b1 - sigma * t.sqrt()) * 0.01;
            rho_call
        },
        "put" => {
            let rho_put = -t * x1 * f64::exp(-r * t) * norm.cdf(-b1 + sigma * t.sqrt()) * 0.01;
            rho_put
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid option type. Use 'call' or 'put'.")),
    };

    Ok(rho)
}

// #################################################################################

/// A Python module implemented in Rust.
#[pymodule]
fn libgai_stocks(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(black_scholes_delta, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_theta, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_vega, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_rho, m)?)?;

    m.add_function(wrap_pyfunction!(bjerksund_stensland_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_theta, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_vega, m)?)?;
    m.add_function(wrap_pyfunction!(bjerksund_stensland_rho, m)?)?;

    Ok(())
}
