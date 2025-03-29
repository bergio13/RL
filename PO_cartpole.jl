using ReinforcementLearning
using Flux
using Plots
using Statistics
using Distributions

# --------------------------
# Policy Network Definition
# --------------------------
struct Policy
    model::Chain
end

# Constructor for Policy
function Policy(s_size::Int, a_size::Int, h_size::Int)
    return Policy(
        Chain(
            Dense(s_size, h_size, relu),
            Dense(h_size, a_size),
            softmax))
end

# Forward pass
(p::Policy)(x) = p.model(x)

# Act function: selects an action using the policy
function act(p::Policy, state)
    probs = p(state)
    d = Categorical(probs)
    action = rand(d)
    return action
end

# --------------------------
# PO Reinforce Algorithm
# --------------------------
function REINFORCE(env, policy, episodes=500; α=0.01, γ=0.99)
    # Initialize optimizer state
    opt_state = Flux.setup(Adam(α), policy.model)

    scores = Float32[]
    avg_scores = Float32[]

    for ep in 1:episodes
        states, actions, rewards = [], [], []

        reset!(env)
        state = Float32.(ReinforcementLearning.state(env))

        done = false
        # Run one episode
        while !done
            push!(states, state)

            # Select action
            action = act(policy, state)
            push!(actions, action)

            # Take step
            ReinforcementLearning.act!(env, action)
            state = Float32.(ReinforcementLearning.state(env))
            reward = ReinforcementLearning.reward(env)
            push!(rewards, reward)
            done = ReinforcementLearning.is_terminated(env)
        end

        # Store rewards for plotting
        push!(scores, sum(rewards))
        push!(avg_scores, mean(scores[max(1, end - 99):end]))

        # Compute discounted returns
        n_steps = length(rewards)
        returns = zeros(Float32, n_steps)
        running_return = 0.0f0
        for t in reverse(1:n_steps)
            running_return = γ * running_return + rewards[t]
            returns[t] = running_return
        end

        # Normalize returns
        μ = mean(returns)
        σ = std(returns) + 1e-8
        normalized_returns = (returns .- μ) ./ σ

        # Compute loss and gradients explicitly
        loss_fn(model) = begin
            l = 0.0f0
            for (s, a, G) in zip(states, actions, normalized_returns)
                p = model(s)
                l -= log(p[a]) * G
            end
            return l
        end

        grads = gradient(loss_fn, policy.model)

        # Update parameters
        Flux.Optimisers.update!(opt_state, policy.model, grads[1])

        # Print progress
        if ep % 50 == 0
            avg = mean(scores[max(1, end - 99):end])
            println("Episode $ep | Reward: $(sum(rewards)) | Avg: $avg")
        end

    end
    return scores, avg_scores
end

#-------
# Function for creating animation
#--------
function create_cartpole_animation(policy, env; max_steps=400)
    states = []
    rewards = Float32[]

    reset!(env)
    state = Float32.(ReinforcementLearning.state(env))
    push!(states, copy(state))

    while true
        action = act(policy, state)
        ReinforcementLearning.act!(env, action)
        push!(rewards, ReinforcementLearning.reward(env))
        state = Float32.(ReinforcementLearning.state(env))
        push!(states, copy(state))
        is_terminated(env) && break
    end

    # Animation parameters
    pole_length = 0.5
    cart_half_width = 0.2

    anim = @animate for (i, s) in enumerate(states)
        cum_reward = sum(rewards[1:min(i - 1, end)])
        x, θ = s[1], s[3]
        pole_x = x + pole_length * sin(θ)
        pole_y = pole_length * cos(θ)

        plot([-2.5, 2.5], [0, 0],
            color=:black, lw=2, legend=false,
            xlim=(-2.5, 2.5), ylim=(-0.5, 1.5),
            aspect_ratio=:equal,
            title="CartPole",
            annotations=[(0.5, 1.4, text("Step: $i", :left, 10))],
            axis=:off)

        plot!([x - cart_half_width, x + cart_half_width], [0, 0],
            lw=8, color=:blue)
        plot!([x, pole_x], [0, pole_y],
            lw=4, color=:red)
    end

    return anim
end

# Set up CartPole environment and parameters
env = CartPoleEnv(; T=Float32, max_steps=400)
action_space_vals = action_space(env)
action_dim = length(action_space_vals)
state_dim = length(ReinforcementLearning.state(env))
h_dim = 32
max_steps = 400

# Instantiate a policy
policy = Policy(state_dim, action_dim, h_dim)
rewards, avg_rewards = REINFORCE(env, policy)

# Create animation
anim = create_cartpole_animation(policy, CartPoleEnv(; T=Float32, max_steps=max_steps))
gif_path = "cartpole_animation.gif"
gif(anim, gif_path, fps=30)

# Plot Results
plot(1:length(rewards), rewards,
    label="Episode Reward",
    xlabel="Episode",
    ylabel="Reward",
    title="REINFORCE on CartPole")
plot!(1:length(avg_rewards), avg_rewards,
    label="Moving Avg (100 eps)",
    linewidth=2, color=:red)
hline!([max_steps], label="Max Reward", ls=:dash)
