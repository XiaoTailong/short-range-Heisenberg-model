using ITensors
using Printf

function calcuate_entropy(psi_e, c)
    orthogonalize!(psi_e, c)
    ## which does not modify the original tensor data
    U,S,V = svd(psi_e[c], (linkind(psi_e, c-1), siteind(psi_e,c)))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN += p^2
    end
    return -log(SvN)
end

using Random
using DataFrames
using DelimitedFiles
using CSV

function bitarr_to_int(arr,s=0)
    v = 1
    for i in view(arr,length(arr):-1:1)
        s += v*i
        v <<= 1
    end 
    s
end

function sample_random_meas(psi_m, NM, N, sites, save_dataframe)
    println("generating samples with randomized measurement")
    NU= 200
    NM = 500
    storage_decimal = zeros(NU,NM)
    # generate the operator for randomized rotation basis
    for i=1:NU
        paulistring = randstring("XYZ", N)
        psi_ = deepcopy(psi_m)  ## no need deepcopy here but deepcopy will be safer
        for j=1:N
            local_unitary = op(string(paulistring[j]), sites[j]) 
            psi_[j] = local_unitary*psi_[j]
            noprime!(psi_[j])
        end
        for k=1:NM
            bits = sample!(psi_).-1
            bits = bits[1:Int(N/2)]
            storage_decimal[i,k] = bitarr_to_int(bits)
        end
    end
    append!(save_dataframe, DataFrame(storage_decimal, :auto))
end


let 
    N = 30
    cutoff = 1E-12
    tau = 0.1
    ttotal = 3

    println("System size:", N)
    # Make an array of 'site' indices
    s = siteinds("S=1/2", N)
    J = rand(N)
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj =
            op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * tau / 2 * hj * J[j])
        push!(gates, Gj)
    end
    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    append!(gates, reverse(gates))

    # Initialize psi to be a product state (alternating up and down)
    psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    c = div(N, 2)
    save_dict = DataFrame()  ## used for save data
    save_Svn = []
    
    # then apply the gates to go to the next time
    for t in 0.0:tau:ttotal
        # SvN = zeros(N-1)  ## used to store the entropy with different partition
        if t==0.5 || t==1.0 || t==3.0
            psi_1 = deepcopy(psi)
            SvN = calcuate_entropy(psi_1, c)  # only record the halfpartition entropy
            println("$t $SvN")
            push!(save_Svn, SvN)
            
            psi_2 = deepcopy(psi)
            
            sample_random_meas(psi_2, 100000, N, s, save_dict)
        end

        t≈ttotal && break

        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end

    CSV.write("./output/save_shadow_N_RM_random$(N)_t$(ttotal).csv", save_dict, header = false)
    writedlm("./output/entropies_N_RM_random$(N)_t$(ttotal).txt", save_Svn)
    # load("save_shadow.jld")["shadow"]
    return 
end