using HuggingFaceApi
using SHA
using UUIDs

const REPOSITORY = "chaitjo/GEOM-DRUGS_ADiT"

const FILES = [
    (
        "processed/geom.pt",
        "geom.pt",
        "9adc3dd2b2a85c1d32bdffde7d3a2cc9eb5e5f1b8407628d0635995bc454d277",
    ),
    (
        "raw/train_smiles.pt",
        "train_smiles.pt",
        "516ecdebf29a5f6bc4782fc83e2d9797515aafc0b601e5c25533a928ff3dc930",
    ),
    (
        "raw/val_smiles.pt",
        "val_smiles.pt",
        "c1c6444236585bd98b50e6ce7dc18e565e4702d4f9d8a7036eda224d46e37c2b",
    ),
    (
        "raw/test_smiles.pt",
        "test_smiles.pt",
        "e2ed79b14113bddc80daa6779c9b10084a8842f276cff61d4942e16c42d7e71f",
    ),
]

function arg(name::String, default::String)
    prefix = "--$name="
    for value in ARGS
        startswith(value, prefix) &&
            return value[(length(prefix) + 1):end]
    end
    return default
end

function sha256_file(path::String)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function download_geom_source(output::String)
    ispath(output) &&
        throw(ArgumentError("output already exists: $output"))
    parent = dirname(output)
    mkpath(parent)
    temporary = joinpath(parent, ".$(basename(output)).tmp-$(uuid4())")
    mkpath(temporary)
    try
        for (remote, local_name, expected_hash) in FILES
            @info "downloading GEOM-Drugs source" remote
            cached = hf_hub_download(
                REPOSITORY,
                remote;
                repo_type="datasets",
            )
            actual_hash = sha256_file(cached)
            actual_hash == expected_hash || throw(ArgumentError(
                "SHA-256 mismatch for $remote: expected $expected_hash, " *
                "received $actual_hash",
            ))
            cp(cached, joinpath(temporary, local_name))
        end
        mv(temporary, output)
        println("wrote=$output files=$(length(FILES))")
    catch
        isdir(temporary) && rm(temporary; recursive=true, force=true)
        rethrow()
    end
end

function main()
    output = abspath(arg(
        "output",
        joinpath(@__DIR__, "..", "..", "data", "geom-source"),
    ))
    download_geom_source(output)
end

main()
