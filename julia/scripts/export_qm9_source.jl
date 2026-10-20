using HuggingFaceApi
using Serialization
using TOML
using UUIDs

const ATOMIC_NUMBERS = Dict(
    'H' => UInt8(1),
    'C' => UInt8(6),
    'N' => UInt8(7),
    'O' => UInt8(8),
    'F' => UInt8(9),
)

function arg(name::String, default::String)
    prefix = "--$name="
    for value in ARGS
        startswith(value, prefix) &&
            return value[(length(prefix) + 1):end]
    end
    return default
end

function write_vector(path::String, values)
    open(path, "w") do io
        write(io, values)
    end
end

function export_qm9_source(output::String)
    ispath(output) &&
        throw(ArgumentError("output already exists: $output"))
    parent = dirname(output)
    mkpath(parent)
    temporary = joinpath(parent, ".$(basename(output)).tmp-$(uuid4())")
    mkpath(temporary)
    try
        source = hf_hub_download(
            "MurrellLab/MGMisc",
            "qm9.jls";
            repo_type="datasets",
        )
        molecules = deserialize(source)
        molecule_count = length(molecules)
        offsets = Vector{Int64}(undef, molecule_count + 1)
        smiles_offsets = Vector{Int64}(undef, molecule_count + 1)
        offsets[1] = 0
        smiles_offsets[1] = 0
        atom_count = 0
        smiles_bytes = 0

        open(joinpath(temporary, "coords.f32"), "w") do coords_io
            open(joinpath(temporary, "atom_number.u8"), "w") do atom_io
                open(joinpath(temporary, "smiles.utf8"), "w") do smiles_io
                    for (index, molecule) in enumerate(molecules)
                        size(molecule.coords, 1) == 3 ||
                            throw(ArgumentError(
                                "molecule $index coordinates are not 3×N",
                            ))
                        size(molecule.coords, 2) == length(molecule.elements) ||
                            throw(ArgumentError(
                                "molecule $index coordinate/element mismatch",
                            ))
                        all(isfinite, molecule.coords) ||
                            throw(ArgumentError(
                                "molecule $index has nonfinite coordinates",
                            ))
                        numbers = UInt8[
                            get(ATOMIC_NUMBERS, element) do
                                throw(ArgumentError(
                                    "unsupported QM9 element $element",
                                ))
                            end
                            for element in molecule.elements
                        ]
                        bytes = codeunits(molecule.smiles)
                        write(coords_io, vec(molecule.coords))
                        write(atom_io, numbers)
                        write(smiles_io, bytes)
                        atom_count += length(numbers)
                        smiles_bytes += length(bytes)
                        offsets[index + 1] = atom_count
                        smiles_offsets[index + 1] = smiles_bytes
                    end
                end
            end
        end
        write_vector(joinpath(temporary, "molecule_offsets.i64"), offsets)
        write_vector(
            joinpath(temporary, "smiles_offsets.i64"),
            smiles_offsets,
        )
        metadata = Dict{String,Any}(
            "format_version" => 1,
            "dataset" => "QM9_MurrellLab_MGMisc",
            "coordinate_units" => "angstrom",
            "molecule_count" => molecule_count,
            "atom_count" => atom_count,
            "source_repository" => "MurrellLab/MGMisc",
            "source_filename" => "qm9.jls",
            "atom_numbers" => sort!(Int.(collect(values(ATOMIC_NUMBERS)))),
        )
        open(joinpath(temporary, "metadata.toml"), "w") do io
            TOML.print(io, metadata; sorted=true)
        end
        mv(temporary, output)
        println(
            "wrote=$output molecules=$molecule_count atoms=$atom_count",
        )
    catch
        isdir(temporary) && rm(temporary; recursive=true, force=true)
        rethrow()
    end
end

function main()
    output = abspath(arg(
        "output",
        joinpath(@__DIR__, "..", "..", "..", "qm9_source"),
    ))
    export_qm9_source(output)
end

main()
