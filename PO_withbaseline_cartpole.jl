using ReinforcementLearning
using Flux
using Plots
using Statistics
using Distributions

# --------------------------
# Network Definitions
# --------------------------
struct Policy
    model::Chain
end

struct ValueNetwork
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

# Constructor for Value Network
function ValueNetwork(s_size::Int, h_size::Int)
    return ValueNetwork(
        Chain(
            Dense(s_size, h_size, relu),
            Dense(h_size, 1)))  # Outputs scalar value estimate
end

# Forward passes
(p::Policy)(x) = p.model(x)
(v::ValueNetwork)(x) = v.model(x)[1]  # Extract scalar

# Act function remains the same
function act(p::Policy, state)
    probs = p(state)
    d = Categorical(probs)
    action = rand(d)
    return action
end

# --------------------------
# REINFORCE with Baseline
# --------------------------
function REINFORCE_with_baseline(env, policy, value_net, episodes=500; α_policy=0.01, α_value=0.01, γ=0.99)
    # Initialize optimizers
    policy_opt = Flux.setup(Adam(α_policy), policy.model)
    value_opt = Flux.setup(Adam(α_value), value_net.model)

    scores = Float32[]
    avg_scores = Float32[]

    for ep in 1:episodes
        states, actions, rewards = [], [], []

        reset!(env)
        state = Float32.(ReinforcementLearning.state(env))
        done = false

        # Run episode
        while !done
            push!(states, state)
            action = act(policy, state)
            push!(actions, action)

            ReinforcementLearning.act!(env, action)
            state = Float32.(ReinforcementLearning.state(env))
            reward = ReinforcementLearning.reward(env)
            push!(rewards, reward)
            done = ReinforcementLearning.is_terminated(env)
        end

        # Store rewards
        total_reward = sum(rewards)
        push!(scores, total_reward)
        push!(avg_scores, mean(scores[max(1, end - 99):end]))

        # Compute discounted returns
        n_steps = length(rewards)
        returns = zeros(Float32, n_steps)
        running_return = 0.0f0
        for t in reverse(1:n_steps)
            running_return = γ * running_return + rewards[t]
            returns[t] = running_return
        end

        # Compute value estimates and advantages
        values = [value_net(s) for s in states]
        advantages = returns .- values

        # Policy gradient update
        policy_grad = gradient(policy.model) do m
            loss = 0.0f0
            for (s, a, A) in zip(states, actions, advantages)
                p = m(s)
                loss -= log(p[a]) * A
            end
            loss
        end
        Flux.update!(policy_opt, policy.model, policy_grad[1])

        # Value function update (MSE loss)
        value_grad = gradient(value_net.model) do m
            loss = 0.0f0
            for (s, R) in zip(states, returns)
                loss += (m(s)[1] - R)^2
            end
            loss
        end
        Flux.update!(value_opt, value_net.model, value_grad[1])

        # Print progress
        if ep % 50 == 0
            avg = mean(scores[max(1, end - 99):end])
            println("Episode $ep | Reward: $total_reward | Avg: $avg")
        end
    end
    return scores, avg_scores
end


# --------------------------
# Training Setup
# --------------------------
max_steps = 400
env = CartPoleEnv(; T=Float32, max_steps=max_steps)
state_dim = length(state(env))
action_dim = length(action_space(env))
h_dim = 64


# Initialize networks
policy = Policy(state_dim, action_dim, h_dim)
value_net = ValueNetwork(state_dim, h_dim)

# Train
scores, avg_scores = REINFORCE_with_baseline(env, policy, value_net, 500)


# Plot results
scatter(scores, label="Episode Reward")
plot!(avg_scores, label="100-episode Average", linewidth=2)
hline!([max_steps], label="Max Reward", linestyle=:dash)
xlabel!("Episode")
ylabel!("Reward")
title!("REINFORCE with Baseline on CartPole")

