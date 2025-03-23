# Parameters
GRID_SIZE = 4
GAMMA = 1.0
THETA = 1e-6

# Actions
ACTIONS = [(0, 1), # Up
    (0, -1), # Down
    (1, 0), # Right
    (-1, 0)] # Left


# Terminal states
TERMINAL_STATES = [(1, 1), (4, 4)]

# Reurns reward for a given state
function reward(state::Tuple{Int,Int})
    if state in TERMINAL_STATES
        return 0.0
    else
        return -1.0
    end
end

# Get next state for a given state and action
function next_state(state::Tuple{Int,Int}, action::Tuple{Int,Int})
    # If in terminal state, return the same state
    if state in TERMINAL_STATES
        return state
    end
    # Else calculate the next state
    next_state = (state .+ action)
    # If next state is out of bounds, return the same state
    if next_state[1] < 1 || next_state[1] > GRID_SIZE || next_state[2] < 1 || next_state[2] > GRID_SIZE
        return state
    end
    return next_state
end

### Value iteration ###
function value_iteration()
    # Initialize policy
    pi = zeros(Int, GRID_SIZE, GRID_SIZE)

    # Initialize value function
    V = zeros(GRID_SIZE, GRID_SIZE)
    new_V = copy(V)

    # Loop until convergence
    while true
        delta = 0.0
        for i in 1:GRID_SIZE
            for j in 1:GRID_SIZE
                state = i, j
                if state in TERMINAL_STATES
                    continue
                end

                # Bellman update
                vals = zeros(length(ACTIONS))
                for (idx, action) in enumerate(ACTIONS)
                    next_s = next_state(state, action)
                    vals[idx] = reward(state) + GAMMA * V[next_s[1], next_s[2]]
                end
                # Update value function
                new_V[i, j] = maximum(vals)

                # Update policy
                pi[i, j] = argmax(vals)

                # Update delta
                delta = max(delta, abs(new_V[i, j] - V[i, j]))
            end
        end
        # Copy new value function to old value function
        V = copy(new_V)

        # Check for convergence
        if delta < THETA
            break
        end
    end

    return V, pi
end


### Policy iteration ###
function policy_iteration()
    V = zeros(GRID_SIZE, GRID_SIZE)
    pi = rand(1:length(ACTIONS), GRID_SIZE, GRID_SIZE)

    while true
        # Policy evaluation
        while true
            delta = 0.0
            new_V = copy(V)

            for i in 1:GRID_SIZE
                for j in 1:GRID_SIZE
                    state = i, j
                    if state in TERMINAL_STATES
                        continue
                    end

                    action = ACTIONS[pi[i, j]]
                    next_s = next_state(state, action)
                    new_V[i, j] = reward(state) + GAMMA * V[next_s[1], next_s[2]]
                    delta = max(delta, abs(new_V[i, j] - V[i, j]))
                end
            end

            V = copy(new_V)

            if delta < 1.01
                break
            end

        end

        # Policy improvement
        policy_stable = true
        for i in 1:GRID_SIZE
            for j in 1:GRID_SIZE
                state = i, j
                if state in TERMINAL_STATES
                    continue
                end

                old_action = pi[i, j]
                vals = zeros(length(ACTIONS))
                for (idx, action) in enumerate(ACTIONS)
                    next_s = next_state(state, action)
                    vals[idx] = reward(state) + GAMMA * V[next_s[1], next_s[2]]
                end

                pi[i, j] = argmax(vals)

                if old_action != pi[i, j]
                    policy_stable = false
                end
            end
        end

        if policy_stable
            break
        end

    end

    return V, pi
end







function main()
    V, pi = value_iteration()
    println("Value iteration:")
    println("Value function:")
    for i in 1:GRID_SIZE
        println(V[i, :])
    end
    println("Policy:")
    for i in 1:GRID_SIZE
        println(pi[i, :])
    end

    V, pi = policy_iteration()
    println("Policy iteration:")
    println("Value function:")
    for i in 1:GRID_SIZE
        println(V[i, :])
    end
    println("Policy:")
    for i in 1:GRID_SIZE
        println(pi[i, :])
    end
end


main()


