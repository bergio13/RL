using Plots, Distributions

# Environment and helper functions 
cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

function draw_card()
    return cards[rand(1:length(cards))]
end

function draw_hand()
    return [draw_card(), draw_card()]
end

function usable_ace(hand)
    return 1 in hand && sum(hand) + 10 <= 21
end

function sum_hand(hand)
    if usable_ace(hand)
        return sum(hand) + 10
    else
        return sum(hand)
    end
end

function is_bust(hand)
    return sum_hand(hand) > 21
end

function dealer_policy(hand)
    while sum_hand(hand) < 17
        push!(hand, draw_card())
    end
    return hand
end

# Function to play a single episode of Blackjack
function play_episode(policy)
    player_hand = draw_hand()
    dealer_hand = draw_hand()
    episode = Vector{Tuple{Tuple{Int,Int,Bool},Union{Int,Nothing}}}()

    # Player's turn
    while true
        current_sum = sum_hand(player_hand)
        dealer_card = dealer_hand[1]

        state = (current_sum, dealer_card, usable_ace(player_hand))

        # Player policy
        action = policy(state)
        push!(episode, (state, action))

        if action == 1  # Hit
            push!(player_hand, draw_card())
            if is_bust(player_hand)
                # Append final state and reward
                final_state = (sum_hand(player_hand), dealer_hand[1], usable_ace(player_hand))
                push!(episode, (final_state, -1))
                return episode
            end
        else  # Stick
            break
        end
    end

    # Dealer's turn
    dealer_hand = dealer_policy(dealer_hand)
    if is_bust(dealer_hand)
        reward = 1
    else
        player_score = sum_hand(player_hand)
        dealer_score = sum_hand(dealer_hand)
        reward = player_score > dealer_score ? 1 : (player_score < dealer_score ? -1 : 0)
    end

    # Append final state and reward
    final_state = (sum_hand(player_hand), dealer_hand[1], usable_ace(player_hand))
    push!(episode, (final_state, reward))
    return episode
end

# On-policy first-visit MC Control for epsilon-soft policies
function mc_control(num_episodes::Int, epsilon::Float64)
    returns_sum = Dict{Tuple{Int,Int,Bool,Int},Float64}()
    returns_count = Dict{Tuple{Int,Int,Bool,Int},Int}()
    Q = Dict{Tuple{Int,Int,Bool,Int},Float64}() # State-action values
    optimal_policy = Dict{Tuple{Int,Int,Bool},Int}() # Policy

    for _ in 1:num_episodes
        # Define epsilon-greedy policy using current Q
        function policy(state::Tuple{Int,Int,Bool})

            actions = [0, 1]  # Possible actions: 0 (stick), 1 (hit)
            num_actions = length(actions)

            # Get Q-values for all state-action pairs
            q_values = [get(Q, (state..., a), 0.0) for a in actions]

            # Find greedy action (break ties randomly)
            max_q = maximum(q_values)
            greedy_actions = findall(q -> q == max_q, q_values) # Find indices of max_q
            greedy_action = actions[rand(greedy_actions)]  # Break ties randomly

            # Construct probability distribution for ε-greedy policy
            action_probs = Dict(a => epsilon / num_actions for a in actions)
            action_probs[greedy_action] += 1 - epsilon  # Boost probability of greedy action

            # Sample action based on probabilities
            chosen_index = rand(Categorical([action_probs[a] for a in actions]))  # Returns index (1 or 2)
            return actions[chosen_index]  # Convert index to action (0 or 1)
        end

        # Generate episode
        episode = play_episode(policy)
        final_reward = episode[end][2]

        # Update Q-values with first-visit MC
        visited = Set{Tuple{Int,Int,Bool,Int}}()
        for step in 1:length(episode)-1
            s, a = episode[step]
            sa = (s..., a)

            if sa ∉ visited
                returns_sum[sa] = get(returns_sum, sa, 0.0) + final_reward
                returns_count[sa] = get(returns_count, sa, 0) + 1
                Q[sa] = returns_sum[sa] / returns_count[sa]
                push!(visited, sa)
            end
        end
    end

    # Derive optimal policy (greedy)
    for sa in keys(Q)
        s, a = (sa[1], sa[2], sa[3]), sa[4]
        if !haskey(optimal_policy, s) || Q[sa] > get(Q, (s..., optimal_policy[s]), -Inf)
            optimal_policy[s] = a
        end
    end

    return Q, optimal_policy
end

function simulate_games(optimal_policy, num_episodes)
    total_reward = 0.0

    # Create policy function from optimal policy dictionary
    function policy(state)
        return get(optimal_policy, state, 0)  # Default to stick (0) if state not found
    end

    for _ in 1:num_episodes
        episode = play_episode(policy)
        final_reward = episode[end][2]  # Get the final reward from the episode
        total_reward += final_reward
    end

    average_reward = total_reward / num_episodes
    return average_reward
end

function detailed_simulation(optimal_policy, num_episodes)
    wins = 0
    losses = 0
    draws = 0

    function policy(state)
        return get(optimal_policy, state, 0)
    end

    for _ in 1:num_episodes
        episode = play_episode(policy)
        final_reward = episode[end][2]

        if final_reward == 1
            wins += 1
        elseif final_reward == -1
            losses += 1
        else
            draws += 1
        end
    end

    println("Results after $num_episodes games:")
    println("Wins:   ", wins, " (", round(100 * wins / num_episodes, digits=1), "%)")
    println("Losses: ", losses, " (", round(100 * losses / num_episodes, digits=1), "%)")
    println("Draws:  ", draws, " (", round(100 * draws / num_episodes, digits=1), "%)")
end

# Plot the optimal policy
function plot_policy(optimal_policy)
    player_sums = 11:21
    dealer_cards = 1:10

    # Prepare policy matrices
    usable_ace_grid = zeros(Int, length(player_sums), length(dealer_cards))
    non_usable_ace_grid = zeros(Int, length(player_sums), length(dealer_cards))

    for (i, p_sum) in enumerate(player_sums)
        for (j, d_card) in enumerate(dealer_cards)
            # Usable Ace
            state_usable = (p_sum, d_card, true)
            usable_ace_grid[i, j] = get(optimal_policy, state_usable, 1)
            # Non-Usable Ace
            state_non_usable = (p_sum, d_card, false)
            non_usable_ace_grid[i, j] = get(optimal_policy, state_non_usable, 1)
        end
    end

    # Plot heatmaps
    p1 = heatmap(dealer_cards, player_sums, usable_ace_grid, title="Usable Ace Policy",
        xlabel="Dealer Card", ylabel="Player Sum", color=:viridis, clim=(0, 1))
    p2 = heatmap(dealer_cards, player_sums, non_usable_ace_grid, title="No Usable Ace Policy",
        xlabel="Dealer Card", ylabel="Player Sum", color=:viridis, clim=(0, 1))
    # Combine and save
    combined_plot = plot(p1, p2, layout=(1, 2), size=(1000, 500))
    #savefig(combined_plot, "blackjack_policy_heatmaps.png")

    # Optionally display the plot
    display(combined_plot)
end


# Run MC Control
@time Q, optimal_policy = mc_control(20_000_000, 0.008)
# Visualize the optimal policy
plot_policy(optimal_policy)

# Play games and evaluate the policy
average = simulate_games(optimal_policy, 100_000)
println("Average reward over 100,000 games: ", round(average, digits=3))
detailed_simulation(optimal_policy, 100_000)