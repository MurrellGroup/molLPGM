using MolLPGMPrep, LinearAlgebra, Printf, Random, TOML
BLAS.set_num_threads(1)
arg(name, default) = String(get(Dict(split(a[3:end], "="; limit=2) for a in ARGS),name,default))

function main()
    domain = arg("domain","qm9")
    corpus = load_corpus(abspath(arg("corpus-dir","data/qm9"));domain)
    output = abspath(arg("out-dir","batches/sampling"))
    ispath(output) && error("Output already exists: $output")
    samples = parse(Int,arg("samples","1024"))
    seed = parse(Int,arg("seed","555921"))
    split = Symbol(arg("split","test"))
    selection = shuffle(MersenneTwister(seed),split_indices(corpus,split))
    0 < samples <= length(selection) || error("Invalid sample count")
    selection = selection[1:samples]
    config = MolecularPrepConfig(; atom_bm_options(arg("atom-bm-profile",""))...)
    Random.seed!(seed)
    rng = MersenneTwister(seed+1)
    mkpath(output)
    first = 1
    batch_index = 0
    while first <= samples
        last = first - 1
        tokens = 0
        while last < samples
            i = selection[last+1]
            n = Int(corpus.conformer_offsets[i+1]-corpus.conformer_offsets[i])
            tokens+n > config.token_capacity && break
            tokens += n
            last += 1
            last-first+1 >= config.max_documents && break
        end
        last >= first || error("Molecule exceeds token capacity")
        batch = make_molecular_training_batch(corpus,selection,config,rng;
                                              selected_indices=selection[first:last])
        batch_index += 1
        path = write_molecular_batch_atomic(output,@sprintf("batch_%06d",batch_index),batch,config;seed,split,domain)
        metadata = TOML.parsefile(joinpath(path,"metadata.toml"))
        metadata["purpose"] = "sampling-only"
        open(io -> TOML.print(io,metadata;sorted=true),joinpath(path,"metadata.toml"),"w")
        first = last+1
        println("wrote=$path atoms=$(batch.n_atoms) documents=$(batch.n_documents)")
    end
    open(joinpath(output,"selection.toml"),"w") do io
        TOML.print(io,Dict("domain"=>domain,"split"=>String(split),"seed"=>seed,
                          "selected_conformer_indices_one_based"=>selection);sorted=true)
    end
end
main()
