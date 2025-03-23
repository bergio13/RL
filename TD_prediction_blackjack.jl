using Plots

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
function play_episode(player_policy_threshold=17)
    player_hand = draw_hand()
    dealer_hand = draw_hand()
    episode = Vector{Tuple{Tuple{Int,Int,Bool},Union{Nothing,Int}}}()

    # Player's turn
    while true
        state = (sum_hand(player_hand), dealer_hand[1], usable_ace(player_hand))
        push!(episode, (state, nothing))

        # Player policy: hit if sum of player hand < than player_policy_threshold
        if sum_hand(player_hand) < player_policy_threshold
            push!(player_hand, draw_card())

            # If player busts, return episode with reward -1
            if is_bust(player_hand)
                push!(episode, ((sum_hand(player_hand), dealer_hand[1], usable_ace(player_hand)), -1))
                return episode
            end
        else
            break # Player sticks
        end
    end

    # Dealer's turn
    dealer = dealer_policy(dealer_hand)
    if is_bust(dealer)
        reward = 1
    else
        player_score = sum_hand(player_hand)
        dealer_score = sum_hand(dealer)
        if player_score > dealer_score
            reward = 1
        elseif player_score < dealer_score
            reward = -1
        else
            reward = 0
        end
    end

    push!(episode, ((sum_hand(player_hand), dealer_hand[1], usable_ace(player_hand)), reward))
    return episode
end

function td_prediction(num_episodes::Int, player_policy_threshold::Int, α::Float64)
    V = Dict{Tuple{Int,Int,Bool},Float64}()  # State value function
    default_value = 0.0  # Initialize unseen states to 0

    for _ in 1:num_episodes
        episode = play_episode(player_policy_threshold)
        for t in 1:length(episode)-1  # Iterate through non-terminal states
            # Current state (s_t) and next state (s_{t+1})
            state = episode[t][1]
            next_state = episode[t+1][1]

            # TD target: R_t + γ * V(s_{t+1})
            # In Blackjack, R_t = 0 for all non-terminal transitions and γ = 1
            target = 0 + get(V, next_state, default_value)

            # TD update: V(s_t) ← V(s_t) + α * [target - V(s_t)]
            current_v = get(V, state, default_value)
            V[state] = current_v + α * (target - current_v)
        end

        # Update the terminal state (last state in the episode)
        terminal_state = episode[end][1]
        terminal_reward = episode[end][2]  # Final reward (win/lose/draw)
        terminal_v = get(V, terminal_state, default_value)
        V[terminal_state] = terminal_v + α * (terminal_reward - terminal_v)
    end
    return V
end

@time V = td_prediction(50_000, 20, 0.01)


# Filter for reasonable player sums
min_player_sum = 12
max_player_sum = 21

# Get state-value estimates for usable and non-usable ace
usable_ace_states = Dict(state => V[state] for state in keys(V) if state[3] && state[1] >= min_player_sum && state[1] <= max_player_sum)
non_usable_ace_states = Dict(state => V[state] for state in keys(V) if !state[3] && state[1] >= min_player_sum && state[1] <= max_player_sum)

# Extract data for usable ace
player_sums_usable = [state[1] for state in keys(usable_ace_states)]
dealer_cards_usable = [state[2] for state in keys(usable_ace_states)]
est_values_usable = [value for value in values(usable_ace_states)]

# Extract data for non-usable ace
player_sums_non_usable = [state[1] for state in keys(non_usable_ace_states)]
dealer_cards_non_usable = [state[2] for state in keys(non_usable_ace_states)]
est_values_non_usable = [value for value in values(non_usable_ace_states)]

# Create 3D scatter plots
p1 = surface(player_sums_usable, dealer_cards_usable, est_values_usable, zcolor=est_values_usable, xlabel="Player sum", ylabel="Dealer card", zlabel="State-value estimate", title="Usable ace: State-value estimates", colorbar=true, legend=false, view=(60, 45))
p2 = surface(player_sums_non_usable, dealer_cards_non_usable, est_values_non_usable, zcolor=est_values_non_usable, xlabel="Player sum", ylabel="Dealer card", zlabel="State-value estimate", title="No usable ace: State-value estimates", colorbar=true, legend=false, view=(60, 45))


# Arrange the plots side by side
plot(p1, p2, layout=(1, 2), size=(900, 400))




