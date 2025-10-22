# Extending Virtual Climate Station Rainfall Records for Auckland

There is a gap in the available rainfall data for Auckland. The Waitakere and Hunua Ranges have long observational records dating back to 1853, while the Virtual Climate Stations (VCSN stations) only extend from 1972 to the present. The short length of the virtual records limits their ability to represent the full range of possible climate behaviour, such as the return intervals of extreme rainfall and drought events that are important for simulating Watercare's catchments of interest. This in turn restricts the modelled characteristic rainfall.

## Aim

The aim of this approach is to extend the VCSN rainfall records using information from the long observational gauges. Rather than resampling the recent record in a purely random manner (i.e. bootstrapping $\overset{\mathrm{iid}}{\sim} N(\mu,\sigma)$), the approach introduces temporal structure from the historical observations that allows the synthetic data to reflect extreme wet and dry periods seen in the long record. This means that extended droughts or exceptional rainfall periods in the historical observations can influence the timing and magnitude of events in the extended virtual series.

## Method

This method operates at the 19-month rolling sum scale to properly capture drought characteristics and long-term rainfall persistence. For each VCSN virtual station (Upper Waikato, Lower Waikato, Waipa River), 19-month rolling rainfall totals are computed from the observed VCSN record (1972-2025). The distribution of these 19-month totals is modeled using a two-component approach where the bulk of the distribution uses empirical resampling from observed values, while the tail is represented by a Generalised Pareto Distribution fitted to exceedances above the 85th percentile threshold.

The cumulative distribution function is defined as:

$$
F_{s}(x) =
\begin{cases}
F_{\text{empirical}}(x), & x \leq u_{s} \\
1 - p_{\text{exceed}}\left(1 + \frac{\xi(x - u_{s})}{\sigma}\right)^{-1/\xi}, & x > u_{s}
\end{cases}
$$

where $F_{\text{empirical}}(x)$ represents direct resampling from the observed non-extreme 19-month totals, $u_s$ is the threshold (85th percentile), and $\sigma$ and $\xi$ are the scale and shape parameters of the GPD. The probability of exceeding the threshold is $p_{\text{exceed}} = \Pr(X > u_{s})$. The use of empirical resampling for the bulk avoids the systematic biases that occur with parametric distributions which tend to over-smooth mid-range values.

Temporal structure is extracted from the Hunua and Waitakere gauge records by computing 19-month rolling sums and calculating an extremeness score for each period based on its percentile rank. This score ranges from 0 to 1 and provides a temporal template of wet and dry periods throughout the historical record. For the extension period (1853-1971), 19-month totals are simulated for each virtual station using the gauge temporal structure as a conditioning variable. The probability of sampling from the extreme tail is modulated by the gauge extremeness through $p_{\text{extreme},t} = p_{\text{exceed}} \times (0.5 + \text{extremeness}_t + \epsilon_t)$, where $\epsilon_t \sim N(0, 0.2^2)$ is random noise that accounts for spatial independence between the gauge sites and virtual stations.

This ensures that when the gauges show extreme wet or dry conditions, all virtual stations have increased probability of extremes, while still sampling from their own site-specific distributions rather than the gauge distributions. The random noise prevents perfect correlation, reflecting the spatial distance between sites, while all three Waikato sites share the same temporal structure to create realistic covariance.

Disaggregation from 19-month to monthly values uses analog pattern resampling. For each simulated 19-month total, similar 19-month periods are found in the VCSN record (within 70-130% of target total, with seasonal preference), and the actual monthly pattern from that VCSN period is resampled and scaled proportionally to match the simulated total exactly. This preserves natural month-to-month variability and seasonal structure. Monthly values are then disaggregated to daily using VCSN daily climatology, calculating the typical within-month distribution of daily rainfall and applying this climatological pattern to the simulated monthly totals. For the VCSN period (1972+), the original daily values are used directly.

## Comparison to Previous Methods

The 2022 iteration used the Stochastic Climate Library (SCL) to generate piecewise stochastic extensions (1853-1906 and 1906-1960) which were then appended to observed records. While this method captured covariance well and produced good return period characteristics, it operated at shorter timescales and did not explicitly condition on the long gauge records. The current approach operates directly at the 19-month scale relevant for droughts, uses explicit conditioning on gauge temporal structure, and employs empirical resampling to avoid distributional assumptions. This produces return period curves that closely match both observed data and the SCL method across all timescales (1 to 1000+ years) while better preserving seasonal variability.

## Limitations

This method assumes that the statistical relationship between the ranges and virtual stations has remained constant through time. Climate change trends are not explicitly modeled in the extension period. Results depend on the choice of threshold for separating bulk from tail (currently 85th percentile), which affects the balance between empirical and parametric modeling. While temporal structure is shared across sites, spatial rainfall patterns and regional weather system dynamics are simplified. The GPD tail extrapolation beyond observed extremes carries inherent uncertainty, as events more extreme than those in the 52-year VCSN record are modeled but not validated. The constraint to match 19-month totals during disaggregation can slightly smooth extreme individual months compared to purely independent sampling.

## References

[1] Carey-Smith, T., et al. (2010). "A comparison of sampling methods for estimating extreme rainfall from a regional climate model." Weather and Climate, 30, 23-48. https://www.metsoc.org.nz/resources/Documents/weather_and_climate/2010_301_23-48_CareySmith.pdf

[2] Bird, D., et al. (2023). "A deep learning approach to downscaling and bias correction of daily precipitation using a weather generator." Geoscientific Model Development, 16, 3785-3812. https://gmd.copernicus.org/articles/16/3785/2023/

[3] Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.

