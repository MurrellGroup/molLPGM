struct MolecularCorpus
    root::String
    metadata::Dict{String,Any}
    coords::Matrix{Float32}
    template::Matrix{Float32}
    atom_class::Vector{UInt8}
    fragment_id::Vector{UInt16}
    conformer_offsets::Vector{Int64}
    source_index::Vector{Int32}
    molecule_id::Vector{Int32}
    conformer_id::Vector{UInt8}
    split::Vector{UInt8}
end

struct MolecularExample
    coords::Matrix{Float32}
    template::Matrix{Float32}
    atom_class::Vector{UInt8}
    fragment_id::Vector{Int32}
    fragment_atom_indices::Vector{Vector{Int}}
    source_index::Int32
    molecule_id::Int32
    conformer_id::UInt8
    split::UInt8
end


function _mmap_vector(path::String, ::Type{T}, count::Int) where {T}
    expected = count * sizeof(T)
    filesize(path) == expected ||
        throw(ArgumentError("$path has $(filesize(path)) bytes; expected $expected"))
    open(path, "r") do io
        return Mmap.mmap(io, Vector{T}, count)
    end
end

function load_corpus(root::String; domain::String="qm9")
    resolved = abspath(root)
    metadata = TOML.parsefile(joinpath(resolved, "metadata.toml"))
    get(metadata, "format_version", 0) == 1 ||
        throw(ArgumentError("unsupported molecular corpus format"))
    dataset = String(get(metadata, "dataset", ""))
    valid_dataset = domain == "qm9" ? startswith(dataset, "QM9") :
                    domain == "geom-drugs" ? startswith(dataset, "GEOM") :
                    throw(ArgumentError("domain must be qm9 or geom-drugs"))
    valid_dataset || throw(ArgumentError(
        "corpus dataset $dataset does not match domain $domain",
    ))
    atom_count = Int(metadata["atom_count"])
    conformer_count = Int(metadata["conformer_count"])
    coords = reshape(
        _mmap_vector(joinpath(resolved, "coords.f32"), Float32, 3 * atom_count),
        3,
        atom_count,
    )
    template = reshape(
        _mmap_vector(joinpath(resolved, "template.f32"), Float32, 3 * atom_count),
        3,
        atom_count,
    )
    corpus = MolecularCorpus(
        resolved,
        metadata,
        coords,
        template,
        _mmap_vector(joinpath(resolved, "atom_class.u8"), UInt8, atom_count),
        _mmap_vector(joinpath(resolved, "fragment_id.u16"), UInt16, atom_count),
        _mmap_vector(
            joinpath(resolved, "conformer_offsets.i64"),
            Int64,
            conformer_count + 1,
        ),
        _mmap_vector(
            joinpath(resolved, "source_index.i32"), Int32, conformer_count,
        ),
        _mmap_vector(
            joinpath(resolved, "molecule_id.i32"), Int32, conformer_count,
        ),
        _mmap_vector(
            joinpath(resolved, "conformer_id.u8"), UInt8, conformer_count,
        ),
        _mmap_vector(joinpath(resolved, "split.u8"), UInt8, conformer_count),
    )
    corpus.conformer_offsets[1] == 0 ||
        throw(ArgumentError("first conformer offset must be zero"))
    corpus.conformer_offsets[end] == atom_count ||
        throw(ArgumentError("final conformer offset does not equal atom count"))
    issorted(corpus.conformer_offsets) ||
        throw(ArgumentError("conformer offsets are not monotone"))
    return corpus
end

function split_indices(corpus::MolecularCorpus, split::Symbol)
    code = split == :train ? UInt8(0) :
           split == :valid ? UInt8(1) :
           split == :test ? UInt8(2) :
           throw(ArgumentError("split must be :train, :valid, or :test"))
    return findall(==(code), corpus.split)
end

function load_example(corpus::MolecularCorpus, conformer_index::Int)
    1 <= conformer_index <= length(corpus.source_index) ||
        throw(BoundsError(corpus.source_index, conformer_index))
    first_atom = Int(corpus.conformer_offsets[conformer_index]) + 1
    final_atom = Int(corpus.conformer_offsets[conformer_index + 1])
    atom_range = first_atom:final_atom
    fragment_id = Int32.(corpus.fragment_id[atom_range])
    nfragments = Int(maximum(fragment_id)) + 1
    atom_indices = [
        findall(==(Int32(fragment)), fragment_id)
        for fragment in 0:(nfragments - 1)
    ]
    all(length(indices) >= Int(corpus.metadata["minimum_fragment_atoms"])
        for indices in atom_indices) ||
        throw(ArgumentError("corpus contains an undersized fragment"))
    return MolecularExample(
        Matrix{Float32}(corpus.coords[:, atom_range]),
        Matrix{Float32}(corpus.template[:, atom_range]),
        Vector{UInt8}(corpus.atom_class[atom_range]),
        fragment_id,
        atom_indices,
        corpus.source_index[conformer_index],
        corpus.molecule_id[conformer_index],
        corpus.conformer_id[conformer_index],
        corpus.split[conformer_index],
    )
end
