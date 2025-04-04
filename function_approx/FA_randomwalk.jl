using Random, Statistics, Plots, Zygote

### Environment
const N_STATES = 1000
const START = 500

function run_episode()
    state = START
    episode = Int[]
    while true
        push!(episode, state)
        step = rand(1:100)
        if rand() < 0.5
            new_state = state - step
            if new_state < 1
                return episode, -1.0
            else
                state = new_state
            end
        else
            new_state = state + step
            if new_state > N_STATES
                return episode, 1.0
            else
                state = new_state
            end
        end
    end
end

### State Aggregation
const N_GROUPS = 10

# Map state to its group index
find_group(s) = ceil(Int, s / 100)


function value_function(s, w)
    group = find_group(s)
    return w[group]
end

### Gradient Monte Carlo with Auto Differentiation
alpha = 2e-4
N_EPISODES = 20_000

# Initialize weights for each group
weights = zeros(N_GROUPS)

# Create an array to count state visits
state_counts = zeros(Int, N_STATES)

for i in 1:N_EPISODES
    episode, reward = run_episode()
    for s in episode
        state_counts[s] += 1
    end

    for s in unique(episode)
        prediction = value_function(s, weights)
        error = reward - prediction

        # Compute the gradient of value_function(s, w) with respect to w using Zygote (even if the gradient would be just 1 if state in group)
        grad = Zygote.gradient(w -> value_function(s, w), weights)[1]

        # Update the weights using the gradient.
        # This is: w = w + α * (G - v̂(s, w)) * ∇_w v̂(s, w)
        weights .= weights .+ alpha * error * grad
    end
end


#### Evaluation
approx_v = [value_function(s, weights) for s in 1:N_STATES]

true_v = range(-1, 1, N_STATES)

# ============= Plotting the Results =============

p1 = plot(1:N_STATES, approx_v,
    label="Approximate Value",
    lw=2,
    xlabel="State",
    ylabel="Value",
    title="State Aggregation Value Function")
plot!(p1, 1:N_STATES, true_v, label="True Value", lw=2, linestyle=:dash)

p2 = histogram(collect(1:N_STATES), weights=state_counts,
    bins=1000,
    xlabel="State",
    ylabel="Frequency",
    title="Distribution of Visited States")

# Display the plots
plot(p1, p2, layout=(2, 1))