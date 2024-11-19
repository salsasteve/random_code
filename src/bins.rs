pub fn compute_octave_bins(num_bins: usize, freq_min: f64, freq_max: f64) -> Vec<(f64, f64)> { 
    let mut bins = Vec::with_capacity(num_bins); 
    let log_min = freq_min.log10(); 
    let log_max = freq_max.log10(); 
    for i in 0..num_bins { 
        let start_freq = 10f64.powf(log_min + (i as f64) * (log_max - log_min) / (num_bins as f64)); 
        let end_freq = 10f64.powf(log_min + ((i + 1) as f64) * (log_max - log_min) / (num_bins as f64)); 
        bins.push((start_freq, end_freq)); 
    } 
        bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_octave_bins() {
        let bins = compute_octave_bins(3, 100.0, 1000.0);
        assert_eq!(bins[0], (100.0, 316.22776601683796));
        assert_eq!(bins[1], (316.22776601683796, 1000.0));
        assert_eq!(bins.len(), 3);
    }


}


