using DelimitedFiles
using PyFormattedStrings
using StatsBase

using Base.Threads

using CUDA

# %%

function treasure_box_kernel(draws::Int, probabilities::CuDeviceVector, probabilities_fever::CuDeviceVector, attempts::CuDeviceVector, rewards::CuDeviceMatrix)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    fever_gauge = 0
    fever_attempts = 2

    while rewards[i, 6] < 1
        roll_fever = rand(Float32)
        
        if roll_fever < 0.005 * draws
            fever_gauge = 110
        end

        if fever_gauge < 110
            for draw in 1:draws
                attempts[i] += 1
                roll = rand(Float32)

                if roll < probabilities[1]
                    @inbounds rewards[i, 1] += 1
                elseif roll < probabilities[2]
                    @inbounds rewards[i, 2] += 1
                elseif roll < probabilities[3]
                    @inbounds rewards[i, 3] += 1
                elseif roll < probabilities[4]
                    @inbounds rewards[i, 4] += 1
                elseif roll < probabilities[5]
                    @inbounds rewards[i, 5] += 1
                elseif roll < probabilities[6]
                    @inbounds rewards[i, 6] += 1
                elseif roll < probabilities[7]
                    @inbounds rewards[i, 7] += 1
                elseif roll < probabilities[8]
                    @inbounds rewards[i, 8] += 1
                elseif roll < probabilities[9]
                    @inbounds rewards[i, 9] += 1
                else
                    @inbounds rewards[i, 10] += 1
                end
            end

            fever_gauge += draws
        else
            for draw in 1:draws
                attempts[i] += 1
                roll = rand(Float32)

                if roll < probabilities_fever[1]
                    @inbounds rewards[i, 1] += 1
                elseif roll < probabilities_fever[2]
                    @inbounds rewards[i, 2] += 1
                elseif roll < probabilities_fever[3]
                    @inbounds rewards[i, 3] += 1
                elseif roll < probabilities_fever[4]
                    @inbounds rewards[i, 4] += 1
                elseif roll < probabilities_fever[5]
                    @inbounds rewards[i, 5] += 1
                elseif roll < probabilities_fever[6]
                    @inbounds rewards[i, 6] += 1
                elseif roll < probabilities_fever[7]
                    @inbounds rewards[i, 7] += 1
                elseif roll < probabilities_fever[8]
                    @inbounds rewards[i, 8] += 1
                elseif roll < probabilities_fever[9]
                    @inbounds rewards[i, 9] += 1
                else
                    @inbounds rewards[i, 10] += 1
                end
            end

            fever_attempts -= 1
            if fever_attempts == 0
                fever_attempts = 2
                fever_gauge = 0    
            end
        end
    end
    
    return
end


function compute_treasure_box(draws::Int, probabilities::Array, probabilities_fever::Array, n::Int, n_threads::Int)
    probabilities = CuArray{Float32}(probabilities)
    probabilities_fever = CuArray{Float32}(probabilities_fever)

    attempts = CUDA.zeros(Int32, n)
    rewards = CUDA.zeros(Int32, n, length(probabilities))

    n_blocks = div(n, n_threads) + Int32(1) * ((n % n_threads) > 0)

    @cuda threads=n_threads blocks=n_blocks treasure_box_kernel(draws, probabilities, probabilities_fever, attempts, rewards)

    return attempts, rewards
end


function quantiles(i::Int, summary::Array, attempts::Array, q)
    if q == "mean"
        summary[i] = mean(attempts)
    else
        summary[i] = attempts[min(ceil(Int, q * attempts.size[1]), attempts.size[1])]
    end

    return
end


function create_summary(attempts::Array)
    q = vcat(["mean"], (1:999)*0.001, [1 - 10^-4, 1 - 10^-5, 1 - 10^-6])
    summary = zeros(Float64, q.size[1])

    attempts = sort(attempts)

    Threads.@threads for i in 1:summary.size[1]
        quantiles(i, summary, attempts, q[i])
    end

    return summary
end


function compute_treasure_box_cases(draws::Int, probabilities::Array, probabilities_fever::Array, n::Int, n_threads::Int)
    attempts_gpu, rewards_gpu = compute_treasure_box(draws, probabilities, probabilities_fever, n, n_threads)                        
    attempts = Array(attempts_gpu)
    rewards = Array(rewards_gpu)

    summary = create_summary(attempts, rewards)

    return summary
end

# %%

draws = 11 # 1
event = false

folder_results = "results"
sub_folder = "treasure_box_sw"
save_path = "./$(folder_results)/$(sub_folder)"

# Approximate number of resets to tune number of attempts
n = 2^26
n_threads = 2^7 # Recommended to be a multiple of 32. Lower this value when you want lower n_magnitude. Maximum depends on GPU

println("-"^50)
println("Number of draws = $(draws)")
println("Event = $(event)")

if event
    # [Chaos, Ancient, Mythic, Legendary + Unique, SW Ancient, SW Mythic, SW Legendary, SW Unique, Chaos x10, Chaos x5]
    # legendary = (4*0.909 + (28-6)*0.520 + 6*0.519) / 100 # 0.1819
    # unique = (4*3.006 + 28*1.717) / 100 # 0.601
    probabilities = cumsum([0, 0.0112, 0.1036, 0.1819 + 0.6010, 
                            0.0003, 0.0003, 0.006, 0.0357, 
                            0.02, 0.04])

    # chaos = (4*0.008 + 28*0.005) / 100 # 0.00172
    # ancient = (4*0.113 + 28*0.064) / 100 # 0.02244
    # mythic = (4*1.036 + 28*0.593) / 100 # 0.20748
    # legendary = (4*2.819 + (28-8)*1.611 + 8*1.610) / 100 # 0.56376
    probabilities_fever = cumsum([0.00172, 0.02244, 0.20748, 0.56376 + 0, 
                                  0.0006, 0.0006, 0.012, 0.0714, 
                                  0.04, 0.08])
else
    probabilities = cumsum([0, 0.0112, 0.1036, 0.18846 + 0.62264, 
                            0.0001, 0.0001, 0.002, 0.0119, 
                            0.02, 0.04])
    probabilities_fever = cumsum([0.00172, 0.02244, 0.20748, 0.62016 + 0, 
                                  0.0002, 0.0002, 0.004, 0.0238, 
                                  0.04, 0.08])
end

attempts_gpu, rewards_gpu = compute_treasure_box(draws, probabilities, probabilities_fever, n, n_threads);                   
attempts = Array(attempts_gpu);
rewards = Array(rewards_gpu);

summary = create_summary(attempts);

if !isdir(save_path)
    mkdir(save_path)
end

writedlm(save_path * "/attempts_$(draws)_$(event).txt", attempts)
writedlm(save_path * "/rewards_$(draws)_$(event).txt", rewards)
writedlm(save_path * "/summary_$(draws)_$(event).txt", summary)