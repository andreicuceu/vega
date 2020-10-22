===============
Config Tutorial
===============

+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|**Parameter Name**       |**Typical Value**             |**Description**                                                                           |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|ap                       |1.0 (usually sampled)         |BAO peak position along the line of sight relative to the fiducial cosmology              |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|at                       |1.0 (usually sampled)         |BAO peak position across the line of sight relative to the fiducial cosmology             |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|aiso                     |1.0 (usually sampled)         |Isotropic BAO peak position relative to the fiducial cosmology. aiso = ap / (1+epsilon)^2 |
|                         |                              |(needed for aiso_epsilon parametrisation)                                                 |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|1+epsilon                |1.0 (usually sampled)         |1+epsilon = aiso / at (needed for aiso_epsilon parametrisation)                           |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|bias_eta_X               |depends on X (usually sampled)|Velocity bias for tracer X. One for each input tracer and all metal lines                 |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|beta_X                   |depends on X (usually sampled |Redshift space distortion parameter (beta = growth_rate * bias_eta / bias_delta)          |
|                         |for LYA and fixed for metals) |for tracer X. One for each input tracer and all metal lines.                              |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|alpha_x                  |depends on X (usually fixed)  |Coefficient for the redshift dependence of the bias of tracer X.  One for each input      |
|                         |                              |tracer and all metal lines                                                                |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|croom_par0               |0.53 (usually fixed)          |First coefficient for the Croom model of quasar bias redshift dependence. See eq. 24 in   |
|                         |                              |the DR14 cross paper (Blomqvist et al 2019).                                              |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|croom_par1               |0.289 (usually fixed)         |Second coefficient for the Croom model of quasar bias redshift dependence. See eq. 24 in  |
|                         |                              |the DR14 cross paper (Blomqvist et al 2019).                                              |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|bias_hcd or              |-0.05 (usually sampled)       |Bias of high column density absorbers. Can be specified as one parameter for all          |
|bias_hcd_X               |                              |components, or as multiple bias_hcd_X, where X is the name of the component e.g. lyaxlya  |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|beta_hcd                 |0.7 (usually sampled)         |Redshift space distortion parameter for high column density.                              |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|L0_hcd                   |depends on HCD model          |Scale parameter for high column density absorbers model. Has different meanings for       |
|                         |(depends on HCD model)        |models. Check the guide on HCD models                                                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|sigmaNL_par              |6.36984 (usually fixed)       |Smoothing parameter for the BAO peak along the line of sight.                             |
|                         |                              |Models large scale non-linear effects                                                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|sigmaNL_per              |3.24 (usually fixed)          |Smoothing parameter for the BAO peak across the line of sight.                            |
|                         |                              |Models large scale non-linear effects                                                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|dnl_arinyo_q1            |0.8558 (usually fixed)        |q1 parameter for the Arinyo model for small scale non-linear effects.                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|dnl_arinyo_kv            |1.11454 (usually fixed)       |kv parameter for the Arinyo model for small scale non-linear effects.                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|dnl_arinyo_av            |0.5378 (usually fixed)        |av parameter for the Arinyo model for small scale non-linear effects.                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|dnl_arinyo_bv            |1.607 (usually fixed)         |bv parameter for the Arinyo model for small scale non-linear effects.                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|dnl_arinyo_kp            |19.47 (usually fixed)         |kp parameter for the Arinyo model for small scale non-linear effects.                     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|par binsize beta_X       |4 (always fixed)              |binsize for the coordinates along the line of sight                                       |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|per binsize beta_X       |4 (always fixed)              |binsize for the coordinates across the line of sight                                      |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|sigma_velo_disp_gauss_X  |(usually sampled)             |gaussian smoothing scale for tracer X to model velocity dispersion (used for QSO)         |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|sigma_velo_disp_lorentz_X|(usually sampled)             |Lorentzian smoothing scale for tracer X to model velocity dispersion (used for QSO)       |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|par_sigma_smooth         |(usually sampled)             |Gaussian smoothing scale for the full correlation function along the line of sight        |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|per_sigma_smooth         |(usually sampled)             |Gaussian smoothing scale for the full correlation function across the line of sight       |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|par_exp_smooth           |(usually sampled)             |Exponential smoothing scale for the full correlation function along the line of sight     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+
|per_exp_smooth           |(usually sampled)             |Exponential smoothing scale for the full correlation function along the line of sight     |
+-------------------------+------------------------------+------------------------------------------------------------------------------------------+