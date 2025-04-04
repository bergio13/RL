using Random, LinearAlgebra, Plots, Statistics

# Environment Constants
const POSITION_MIN = -1.2
const POSITION_MAX = 0.5
const VELOCITY_MIN = -0.07
const VELOCITY_MAX = 0.07
const ACTIONS = [-1, 0, 1]  # Reverse, Neutral, Forward
const ALPHA = 0.2 / 8       # Step size for weight update
const GAMMA = 1.0           # Discount factor
const EPSILON = 0.001         # Epsilon for ε-greedy policy

mutable struct MountainCarEnv
    position::Float64
    velocity::Float64
end

function reset!(env::MountainCarEnv)
    env.position = rand() * 0.2 - 0.6  # Random start in [-0.6, -0.4)
    env.velocity = 0.0
    return env.position, env.velocity
end

function step!(env::MountainCarEnv, action::Int)
    force = 0.001 * ACTIONS[action]
    gravity = -0.0025 * cos(3 * env.position)
    env.velocity = clamp(env.velocity + force + gravity, VELOCITY_MIN, VELOCITY_MAX)
    env.position = clamp(env.position + env.velocity, POSITION_MIN, POSITION_MAX)

    if env.position == POSITION_MIN
        env.velocity = 0.0  # Reset velocity if hitting left bound
    end

    reward = -1  # Reward is always -1 until reaching the goal
    done = env.position >= POSITION_MAX
    return env.position, env.velocity, reward, done
end

# Tile Coding Parameters
const NUM_TILINGS = 8
const NUM_TILES_POS = 8
const NUM_TILES_VEL = 8
const TOTAL_FEATURES = NUM_TILINGS * NUM_TILES_POS * NUM_TILES_VEL

const TILE_WIDTH_POS = (POSITION_MAX - POSITION_MIN) / (NUM_TILES_POS - 1)
const TILE_WIDTH_VEL = (VELOCITY_MAX - VELOCITY_MIN) / (NUM_TILES_VEL - 1)

"""
    tile_coding(position, velocity)

Returns a binary feature vector of length TOTAL_FEATURES with one active feature per tiling.
Each tiling is offset to provide overlapping grids.
"""
function tile_coding(position::Float64, velocity::Float64)
    features = zeros(Int, TOTAL_FEATURES)
    for tiling in 1:NUM_TILINGS
        # Compute offsets for this tiling
        offset_pos = (tiling - 1) * TILE_WIDTH_POS / NUM_TILINGS
        offset_vel = (tiling - 1) * TILE_WIDTH_VEL / NUM_TILINGS

        pos_index = floor(Int, (position - POSITION_MIN + offset_pos) / TILE_WIDTH_POS)
        vel_index = floor(Int, (velocity - VELOCITY_MIN + offset_vel) / TILE_WIDTH_VEL)

        # Clamp indices to valid range
        pos_index = clamp(pos_index, 0, NUM_TILES_POS - 1)
        vel_index = clamp(vel_index, 0, NUM_TILES_VEL - 1)

        # Compute the index for this tiling.
        # Each tiling uses a block of NUM_TILES_POS * NUM_TILES_VEL features.
        tiling_offset = (tiling - 1) * (NUM_TILES_POS * NUM_TILES_VEL)
        tile_index = tiling_offset + pos_index * NUM_TILES_VEL + vel_index + 1  # +1 for 1-indexing
        features[tile_index] = 1
    end
    return features
end

function q_value(w, features, action)
    return dot(features, w[:, action])
end

function epsilon_greedy(w, features, epsilon)
    if rand() < epsilon
        return rand(1:length(ACTIONS))  # Random action
    else
        return argmax([q_value(w, features, a) for a in 1:length(ACTIONS)])
    end
end

function semi_gradient_sarsa!(env, episodes)
    # Initialize Weights: one weight per feature per action
    w = zeros(TOTAL_FEATURES, length(ACTIONS))

    for episode in 1:episodes
        reset!(env)
        features = tile_coding(env.position, env.velocity)
        action = epsilon_greedy(w, features, EPSILON)
        done = false
        while !done
            position, velocity, reward, done = step!(env, action)
            next_features = tile_coding(position, velocity)
            next_action = epsilon_greedy(w, next_features, EPSILON)

            target = reward + GAMMA * q_value(w, next_features, next_action) * (!done)
            error = target - q_value(w, features, action)
            w[:, action] .+= ALPHA * error .* features

            features = next_features
            action = next_action
        end
    end
    return w
end

# Training and Evaluation
function evaluate_policy(env, w, episodes)
    episode_lengths = Int[]
    for _ in 1:episodes
        reset!(env)
        done = false
        steps = 0
        while !done && steps <= 1000
            features = tile_coding(env.position, env.velocity)
            action = epsilon_greedy(w, features, 0.0)  # purely greedy
            _, _, _, done = step!(env, action)
            steps += 1
        end
        push!(episode_lengths, steps)
    end
    return episode_lengths
end

# Run Training
env = MountainCarEnv(0.0, 0.0)
trained_w = semi_gradient_sarsa!(env, 1_000)

# Evaluate Policy
episode_lengths = evaluate_policy(env, trained_w, 200)

# Calculate and print mean episode length
println("Mean episode length: ", mean(episode_lengths))
println("Standard deviation of episode length:", std(episode_lengths))


# Learning Curve Visualization
function learning_curve(n_episodes::Int, n_runs::Int, increment::Int)
    all_average_steps = zeros(length(1:increment:n_episodes), n_runs)

    for run in 1:n_runs
        curve_index = 1
        average_steps_per_episode = Float64[]

        for i in 1:increment:n_episodes
            env = MountainCarEnv(0.0, 0.0)
            w = semi_gradient_sarsa!(env, i)
            episode_lengths = evaluate_policy(env, w, 200)  # Increased eval episodes
            push!(average_steps_per_episode, mean(episode_lengths))

            all_average_steps[curve_index, run] = mean(episode_lengths)
            curve_index += 1
        end
    end

    # Compute mean and standard deviation across runs
    mean_steps = vec(mean(all_average_steps, dims=2))
    std_steps = vec(std(all_average_steps, dims=2))

    return 1:increment:n_episodes, mean_steps, std_steps
end

# Run multiple experiments
n_episodes = 500
increment = 10
n_runs = 5  # Number of independent runs

x, mean_steps, std_steps = learning_curve(n_episodes, n_runs, increment)

# Plot with error bars
p = plot(x, mean_steps,
    ribbon=std_steps,  # Add error bars
    fillalpha=0.2,    # Transparency of error band
    label="Average Steps to Goal",
    xlabel="Training Episodes",
    ylabel="Average Steps to Goal",
    title="Stable Learning Curve (SARSA with Multiple Runs)",
    legend=:topright)

display(p)


function visualize_policy_space(env::MountainCarEnv, w, max_steps::Int=200)
    # Reset the environment
    reset!(env)

    # Storage for positions and velocities
    positions = Float64[env.position]
    velocities = Float64[env.velocity]

    # Run one episode
    done = false
    steps = 0

    while !done && steps < max_steps
        # Get current state features
        features = tile_coding(env.position, env.velocity)

        # Choose action greedily
        action = epsilon_greedy(w, features, 0.0)

        # Take a step
        position, velocity, reward, done = step!(env, action)

        # Store results
        push!(positions, position)
        push!(velocities, velocity)

        steps += 1
    end

    return positions, velocities, steps
end

# Train the agent
env = MountainCarEnv(0.0, 0.0)
trained_w = semi_gradient_sarsa!(env, 1_000)

# Visualize the policy
reset!(env)
positions, velocities, steps = visualize_policy_space(env, trained_w)

## Create animation
anim = @animate for i in eachindex(positions)
    # Two subplots
    p1 = plot(title="Phase Space",
        xlabel="Position",
        ylabel="Velocity",
        xlims=(POSITION_MIN, POSITION_MAX),
        ylims=(VELOCITY_MIN, VELOCITY_MAX),
        legend=false)


    # Plot trajectory in state space
    plot!(p1, positions[1:i], velocities[1:i],
        color=:blue,
        label="Trajectory",
        linewidth=0.5)

    # Highlight current position
    scatter!(p1, [positions[i]], [velocities[i]],
        color=:red,
        markersize=5)

    # Second subplot for mountain curve visualization
    p2 = plot(
        xlims=(POSITION_MIN, POSITION_MAX),
        ylims=(-1.2, 1),
        legend=false
    )
    # Mountain curve with current position
    x = range(POSITION_MIN, POSITION_MAX, length=100)
    y = sin.(3 * x)
    plot!(p2, x, y)

    scatter!(p2, [positions[i]], [sin.(3 * positions[i])],
        color=:red,
        markersize=5,
        label="Car Position")

    # Combine plots
    plot(p1, p2, layout=(1, 2), size=(1000, 400))
end

# Save the animation
gif(anim, "mountain_car_movement.gif", fps=10)

println("Animation saved as mountain_car_movement.gif")
println("Total steps taken: ", steps)

