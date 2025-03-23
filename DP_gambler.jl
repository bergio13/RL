using Random, OffsetArrays, Plots

function value_iteration(ph=0.4, Theta=1e-6)
    # Create a value array V with random initial values
    V = OffsetArray(rand(100) .* 10, 0:99)
    V[0] = 0.0  # set state 0 to zero

    # Create a policy array pi 
    pi = OffsetArray(zeros(100), 0:99)

    counter = 1
    while true
        Delta = 0.0
        # Loop over states 1 to 99 (Python range(1, 100))
        for s in 1:99
            old_v = V[s]
            max_a = min(s, 100 - s)  # maximum allowed bet for state s
            # Create an action value array with indices 1 to max_a
            v = OffsetArray(zeros(max_a), 1:max_a)
            for a in 1:max_a
                # Calculate the value of each action
                if s + a < 100 # if not at the goal state
                    v[a] += ph * (0 + V[s+a]) + # Winning bet (no reward, just future value)
                            (1 - ph) * (0 + V[s-a]) # Losing bet (no reward, just future value)
                elseif s + a == 100 # if at the goal state
                    v[a] += ph + # Winning bet (reward)
                            (1 - ph) * V[s-a] # Losing bet (no reward, just future value)
                end
            end
            # Find the optimal action (argmax) and its corresponding value
            max_val, op_a = findmax(v)
            pi[s] = op_a
            V[s] = max_val
            Delta = max(Delta, abs(old_v - V[s]))
        end

        counter += 1
        if counter % 1000 == 0
            println("train loop ", counter)
            println("Delta = ", Delta)
        end
        if Delta < Theta
            break
        end
    end

    # Return the values and policies for states 1 through 99
    V_result = [V[s] for s in 1:99]
    pi_result = [pi[s] for s in 1:99]
    return V_result, pi_result
end

function main()
    V1, pi1 = value_iteration(0.4)
    V2, pi2 = value_iteration(0.25)
    V3, pi3 = value_iteration(0.55)

    S = 1:99  # state labels for plotting

    # Plot value functions for different ph values
    plt1 = plot(S, V1, label="ph=0.4", lw=2, title="Value Functions", xlabel="State", ylabel="Value")
    plot!(plt1, S, V2, label="ph=0.25", lw=2)
    plot!(plt1, S, V3, label="ph=0.55", lw=2)
    display(plt1)

    # Plot the policies as bar charts
    plt2 = bar(S, pi1, title="Policy for ph=0.4", xlabel="State", ylabel="Optimal Bet")
    display(plt2)
    plt3 = bar(S, pi2, title="Policy for ph=0.25", xlabel="State", ylabel="Optimal Bet")
    display(plt3)
    plt4 = bar(S, pi3, title="Policy for ph=0.55", xlabel="State", ylabel="Optimal Bet")
    display(plt4)
end

main()
