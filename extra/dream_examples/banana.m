MCMCPar = struct()
Extra = struct()
Measurement = struct()
ParRange = struct()

MCMCPar.n = 10;                         # Dimension of the problem (Nr. parameters to be optimized in the model)
MCMCPar.seq = MCMCPar.n;                # Number of Markov Chains / sequences
MCMCPar.DEpairs = 3;                    # Number of chain pairs to generate candidate points
MCMCPar.Gamma = 0;                      # Kurtosis parameter Bayesian Inference Scheme
MCMCPar.nCR = 3;                        # Crossover values used to generate proposals (geometric series)
MCMCPar.ndraw = 10000;                 # Maximum number of function evaluations
MCMCPar.steps = 10;                     # Number of steps
MCMCPar.eps = 5e-2;                     # Random error for ergodicity
MCMCPar.outlierTest = 'IQR_test';       # What kind of test to detect outlier chains?

# -----------------------------------------------------------------------------------------------------------------------
Extra.pCR = 'Update';                   # Adaptive tuning of crossover values
# -----------------------------------------------------------------------------------------------------------------------

# --------------------------------------- Added for reduced sample storage ----------------------------------------------
Extra.reduced_sample_collection = 'Yes';# Thinned sample collection?
Extra.T = 10;                           # Every Tth sample is collected
# -----------------------------------------------------------------------------------------------------------------------

# Define the specific properties of the banana function
Extra.mu   = zeros([1,MCMCPar.n]);                      # Center of the banana function
Extra.cmat = eye(MCMCPar.n); Extra.cmat[0,0] = 100;     # Target covariance
Extra.imat = inv(Extra.cmat);                           # Inverse of target covariance
Extra.bpar = 0.1;                                       # "bananity" of the target, see bananafun.m

# What type of initial sampling
Extra.InitPopulation = 'COV_BASED';
# Provide information to do alternative sampling
Extra.muX = Extra.mu;                                   # Provide mean of initial sample
Extra.qcov = eye(MCMCPar.n) * 5;                        # Initial covariance
# Save all information in memory or not?
Extra.save_in_memory = 'No';

# Give the parameter ranges (minimum and maximum values)
ParRange.minn = -Inf * ones([1,MCMCPar.n]); ParRange.maxn = Inf * ones([1,MCMCPar.n]);

# Define the boundary handling
Extra.BoundHandling = 'None';
# Define data structures for use in computation of posterior density
Measurement.MeasData = []; Measurement.Sigma = []; Measurement.N = 0;
# Define modelName
ModelName = 'Banshp';
# Define likelihood function
option = 4;

[Sequences,Reduced_Seq,X,output,hist_logp] = dream(MCMCPar,ParRange,Measurement,ModelName,Extra,option)
