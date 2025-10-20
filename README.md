# Extending Virtual Climate Station Rainfall Records for Auckland

There is a  in the available rainfall data for Auckland. The Waitakere and Hunua Ranges have long observational records dating back to the 1853, while the Virtual Climate Stations (VCSN stations), only extend from 1972 to the present. The short length of the virtual records limits their ability to represent the full range of possible climate behaviour, such as the return intervals of extreme rainfall and drought events that are important for simulating Watercare's catchments of interest. This in turn restricts the modelled characteristic rainfall.

## Aim

The aim of this proposal is to extend the VCSN rainfall records using information from the long observational gauges. Rather than resampling the recent record in a purely random manner (i.e. bootstrapping $\overset{\mathrm{iid}}{\sim} N(\mu,\sigma)$), the approach introduces conditional probability relationships that allow the synthetic data to reflect the behaviour seen in the historical observations. This means that months with exceptional rainfall in the long record can influence the probability and magnitude of extreme events in the extended virtual series. In this way the synthetic data retain realistic temporal structure while maintaining the structure of extreme events that may not appear in the short modern record E.g. 1930s droughts.

## Method

In the 2022 iteration of this method, the Waikato sites were stochastically generated using the Stochastic Climate Library (SCL library):

> The observed record from the Waikato sites only extended back to 1960 and therefore the length of this observed record (62 years) was shorter than the observed records in the Hunua and Waitakere Ranges (168 years). To generate co-variant stochastics across Auckland and the Waikato, the Waikato record fed into SCL needed to be extended by 106 years back to 1853 to match the Auckland records. A piece-wise dataset for Waikato was created through appending two stochastic iterations to the observed record. To do this, SCL was used to generate 10 iterations of stochastic data for Lower Waikato between 1853 and 1960 (broken up into two periods of 1853-1906 and 1906-1960).

To extend the VCSN records, rainfall distributions are separated into common and extreme ranges, with the upper tail represented by a Generalised Pareto Distribution. Dependence between the long gauges and the virtual stations is described through statistical models that link both the occurrence and the size of extreme events. This couples the probability of an extreme month at the virtual Waikato stations to vary in line with the state of the observational gauges. For the other, non extreme months, resampling methods preserve the typical ebb & flow that is consistent with normal rainfall patterns. 

For each station $ s $ and calendar month $ m $, the rainfall values $ X_{s,m} = \{x_{s,m,t}\} $ are separated by a threshold $ u_{s,m} $, at some percentile of that month’s distribution. The model for the cumulative distribution function $ F_{s,m}(x) $ is defined as

$$
F_{s,m}(x) =
\begin{cases}
F_{\text{bulk},s,m}(x), & x \leq u_{s,m} \\
1 - p_{\text{exceed}}\left(1 + \frac{\xi(x - u_{s,m})}{\sigma}\right)^{-1/\xi}, & x > u_{s,m}
\end{cases}
$$

where $F_{\text{bulk},s,m}(x)$ is the empirical distribution of non-extreme rainfall, $ p_{\text{exceed}} = \Pr(X > u_{s,m}) $, and $ \sigma $ and $ \xi $ are the scale and shape parameters of the Generalised Pareto Distribution fitted to the exceedances. Dependence between the observed gauges and the virtual stations is introduced through a conditional probability model that links both the occurrence and the magnitude of extreme rainfall. The probability of an exceedance at a virtual station is defined as

$$
\Pr(I_{v,m,t} = 1 \mid I_{g,m,t}, Z_t) = \text{logit}^{-1}(\alpha_m + \beta_m I_{g,m,t} + \gamma_m Z_t)
$$

where $ I_{v,m,t} $ and $ I_{g,m,t} $ are exceedance indicators for the virtual and observed gauge stations respectively, and $ Z_t $ represents optional climate covariates such as the ENSO index. When an exceedance occurs, the rainfall magnitude is drawn from the conditional tail model, ensuring that extreme events at the virtual station are consistent with those observed at the ranges. For months that are not extreme, a block-resampling technique is used to maintain realistic persistence and seasonal variation.


This process produces long synthetic records that are consistent with both the statistical characteristics of the observed data and the physical relationship between the ranges and the lower catchments.

This allows for multiple stochastic realisations which can be used to assess uncertainty in long term rainfall behaviour. It also gives a structured way to integrate short modelled records with long observed ones, ensuring that the information contained in historical data is not lost.

## Literature

This method is well-supported by literature. For example, Carey-Smith et al. (2010) [1] use a generalised Pareto distribution (GPD) from selected extreme events using a mean + 2 S.D. threshold. This paper also verifies that, at these extremes, it is reasonable to consider consecutive events as independent samples (where correlation < .07).

Furthermore, it is very common and standardised for stochastic weather generators to use this piecewise distribution, where one captures the mean reverting behaviour of day-to-day weather and another (GPD in this proposed method) for sampling extremes. In literature, this is refered to as a ``two-stage model''. Modern stochastic weather generators make use of this, such as Bird et al. (2023) [2] which, while using a deep learning approach, are able to succesfully reproduce observed seasonality while 

## Limitations

Obviously, this method carries its own limitations. One is that is assumes that the relationship between the ranges and the virtual stations has remained constant through time. While in the past stationarity has been modelled, this inter-region stationarity was not. 

Furthermore, this method is sensitive on the choice of thresholds for defining extremes and on the accuracy of fitted distributions, both of which can influence the resulting statistics. 

While conditional sampling captures many aspects of dependence, it does not fully reproduce spatial rainfall patterns or system dynamics (e.g. storm in the Waikato, dry in Auckland) that are captured by the virtual records.

## References

[1] https://www.metsoc.org.nz/resources/Documents/weather_and_climate/2010_301_23-48_CareySmith.pdf#

[2] https://gmd.copernicus.org/articles/16/3785/2023/#

