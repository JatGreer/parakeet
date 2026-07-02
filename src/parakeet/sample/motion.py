import numpy as np
from scipy.spatial import cKDTree

def update_particle_position_and_direction(
    position,
    direction,
    global_drift,
    interaction_range,
    velocity,
    noise_magnitude,
    fraction_number,
):
    """
    Update the particle positions and directions using the Vicsek model

    Params:
        position: An array of particle positions (A)
        direction: An array of particle directions (radians)
        global_drift: The global drift (A/time)
        interaction_range: The distance at which particles interact (A)
        velocity: The constant velocity of the system (A/time)
        noise_magnitude: The direction noise magnitude (radians)

    Returns:
        (position, direction) of the particles

    """

    # the full model looks like this:
    # r_(i+1) = r_i + z_i + w_i
    # where z_i is the random Brownian Motion and is sampled from N(0, sqrt(2*D*dt))
    # w_i is the directed motion component, similar to the aTrack model but with time-dependent magnitude as well:
    # w_i = v_i [cos(theta_i), sin(theta_i)]
    # here, v_i is the drift velocity at frame i, which is given by:
    # v_i = v_0 exp(-(i*dt)/tau)
    # v_0 and tau are parameters not estimated by aTrack
    # the direction of the drift theta_i is also subject to diffusion as in the aTrack model:
    # theta_(i+1) = theta_i + dtheta
    # with the change in direction given by:
    # dtheta ~ N(0, sqrt(2*D_ang*dt)C)
    # where D_ang is the angular diffusion coefficient estimated by aTrack and C is the covariance matrix that introduces spatial correlation between the tracks
    # C_ij = exp(-|r_i - r_j|/lambda)
    # where lambda is the correlation length, for which we use the estimated value from RELION's polishing as a starting point

    # this gives 5 parameters, v0, tau, D, D_ang and lambda
    # We're using the old variable names for now, so
    # global_drift -- > [D, D_ang]
    # interaction_range -- > lambda
    # velocity --> v_0
    # noise_magnitude -- > tau

    # The time step. In practice this should be equal to the exposure time, or time between
    # frames, but for now we'll keep it such that the total exposure time is 10 s for 50 frames
    # i.e. dt = 0.2 s
    dt = 0.04 # in seconds
    D = global_drift[0]
    D_ang = global_drift[1]
    v0 = velocity
    tau = noise_magnitude
    nb_sub_steps = 10 # computing the update in 100 sub steps to make it smoother
    sub_dt = dt / nb_sub_steps

    # First compute the matrix C and its Cholesky factorization. This is used to update the direction of the drift
    N = position.shape[0] # number of particles
    distance = cKDTree(position[:,:2]).sparse_distance_matrix(cKDTree(position[:,:2]), max_distance=np.inf).toarray()
    C = np.exp(-distance / interaction_range)
    C_cholesky = np.linalg.cholesky(C + 1e-7 * np.eye(N))

    # if it is the first frame, we need to sample the initial directions. These should be the same for all particles
    if fraction_number == 0:
        theta0 = np.random.rand()*2*np.pi

    new_positions = np.zeros((N, nb_sub_steps+1, 2))
    new_positions[:,0,:] = position[:,:2]
    for tt in range(nb_sub_steps):
        # Now compute the drift velocity at the current frame
        t = fraction_number*dt + tt*sub_dt
        vel = v0 * np.exp( - t / tau) # only the magnitude

        dtheta = np.random.randn(N)
        dtheta = np.dot(C_cholesky, dtheta) * np.sqrt(2 * D_ang * sub_dt)
        # dtheta = dtheta * np.sqrt(2 * D_ang * sub_dt)

        if tt == 0 and fraction_number == 0:
            theta = np.ones(N) * theta0 + dtheta
        elif tt == 0:
            theta = direction + dtheta
        else:
            theta = theta + dtheta

        vel = vel / nb_sub_steps * np.stack([np.cos(theta), np.sin(theta)], axis=1) # shape (N, 2)
        # print(vel.shape)

        # Finally, sample the Brownian Motion
        xi = np.random.normal(0, np.sqrt(2 * D * sub_dt), (N,2)) # shape (N, 2)

        new_positions[:,tt+1,:] = new_positions[:,tt,:] + vel + xi

    final_positions = new_positions[:,-1,:]
    final_positions = np.hstack([final_positions, position[:,2:]]) # keep z and other dimensions unchanged

    # Return the position and direction
    return final_positions, theta
