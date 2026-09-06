# Differential test of overtone-sim against Yao.jl.
#
# Reads the cases emitted by
#   cargo run -p overtone-sim --example emit_oracle_cases > lab/test/cases.txt
# rebuilds each circuit in Yao, and checks the expectation value and the gradient.
#
# Two independent Yao-side gradients are computed, not one:
#
#   1. Yao's own reverse-mode autodiff, via expect'.
#   2. A parameter-shift gradient evaluated with Yao as the state-vector engine.
#
# If those two disagree with each other, the fault is in how this script drives Yao --
# most likely the ordering of `parameters(circuit)` -- rather than in the Rust engine.
# Checking both means a red test points at the right place.
#
# Conventions, verified rather than assumed:
#   - Yao's Rx(t) is exp(-i t X / 2), matching overtone-sim.
#   - Yao is little-endian, and Yao qubit q+1 corresponds to overtone-sim qubit q.

using Yao
using Test
using Printf

const TOL = 1e-12

struct GateSpec
    kind::Symbol           # :RX :RY :RZ :H :CZ :CX
    a::Int                 # qubit, or control
    b::Int                 # target, for two-qubit gates
    param_index::Int       # 0-based index into params, or -1 for a fixed angle
    scale::Float64         # chain-rule factor: angle = scale * params[index]
    fixed::Float64         # the angle, when param_index == -1
end

struct Case
    id::Int
    n::Int
    nparams::Int
    params::Vector{Float64}
    gates::Vector{GateSpec}
    obs::String
    value::Float64
    grad::Vector{Float64}
end

function parse_cases(path::AbstractString)
    cases = Case[]
    id = 0; n = 0; nparams = 0
    params = Float64[]
    gates = GateSpec[]
    obs = ""; value = 0.0
    grad = Float64[]

    for line in eachline(path)
        isempty(strip(line)) && continue
        startswith(line, "#") && continue
        parts = split(line)

        if parts[1] == "CASE"
            id = parse(Int, parts[2]); n = parse(Int, parts[3]); nparams = parse(Int, parts[4])
            gates = GateSpec[]
        elseif parts[1] == "PARAMS"
            params = [parse(Float64, x) for x in parts[2:end]]
        elseif parts[1] == "G"
            kind = Symbol(parts[2])
            if kind in (:RX, :RY, :RZ)
                q = parse(Int, parts[3])
                if parts[4] == "P"
                    push!(gates, GateSpec(kind, q, -1, parse(Int, parts[5]),
                                          parse(Float64, parts[6]), 0.0))
                else
                    push!(gates, GateSpec(kind, q, -1, -1, 1.0, parse(Float64, parts[5])))
                end
            elseif kind == :H
                push!(gates, GateSpec(kind, parse(Int, parts[3]), -1, -1, 1.0, 0.0))
            else  # CZ, CX
                push!(gates, GateSpec(kind, parse(Int, parts[3]), parse(Int, parts[4]),
                                      -1, 1.0, 0.0))
            end
        elseif parts[1] == "OBS"
            obs = parts[2]
        elseif parts[1] == "VALUE"
            value = parse(Float64, parts[2])
        elseif parts[1] == "GRAD"
            grad = length(parts) > 1 ? [parse(Float64, x) for x in parts[2:end]] : Float64[]
        elseif parts[1] == "ENDCASE"
            push!(cases, Case(id, n, nparams, params, gates, obs, value, grad))
        end
    end
    cases
end

"The angle a gate rotates by, given the parameter vector."
function angle_of(g::GateSpec, params::Vector{Float64})
    g.param_index < 0 ? g.fixed : g.scale * params[g.param_index + 1]
end

"Build the Yao block for one gate. `offsets` displaces one gate's angle, for the shift rule."
function yao_gate(g::GateSpec, n::Int, params::Vector{Float64}, offset::Float64)
    # overtone-sim qubit q is Yao qubit q+1.
    if g.kind == :RX
        put(n, g.a + 1 => Rx(angle_of(g, params) + offset))
    elseif g.kind == :RY
        put(n, g.a + 1 => Ry(angle_of(g, params) + offset))
    elseif g.kind == :RZ
        put(n, g.a + 1 => Rz(angle_of(g, params) + offset))
    elseif g.kind == :H
        put(n, g.a + 1 => H)
    elseif g.kind == :CZ
        control(n, g.a + 1, g.b + 1 => Z)
    elseif g.kind == :CX
        control(n, g.a + 1, g.b + 1 => X)
    else
        error("unknown gate kind $(g.kind)")
    end
end

"The whole circuit, optionally with gate `shift_index` displaced by `shift`."
function yao_circuit(c::Case; shift_index::Int = -1, shift::Float64 = 0.0)
    blocks = [yao_gate(g, c.n, c.params, i == shift_index ? shift : 0.0)
              for (i, g) in enumerate(c.gates)]
    chain(c.n, blocks...)
end

function yao_observable(c::Case)
    c.obs == "Z0" ? put(c.n, 1 => Z) : kron(c.n, (i => Z for i in 1:c.n)...)
end

"Expectation value from Yao."
function yao_value(c::Case; shift_index::Int = -1, shift::Float64 = 0.0)
    reg = zero_state(c.n)
    reg |> yao_circuit(c; shift_index = shift_index, shift = shift)
    real(expect(yao_observable(c), reg))
end

"Gradient by parameter shift, using Yao only as the state-vector engine."
function yao_shift_grad(c::Case)
    grad = zeros(Float64, c.nparams)
    for (i, g) in enumerate(c.gates)
        g.param_index < 0 && continue
        plus  = yao_value(c; shift_index = i, shift =  pi / 2)
        minus = yao_value(c; shift_index = i, shift = -pi / 2)
        # The shift gives d/d(angle); the chain rule converts to d/d(parameter).
        grad[g.param_index + 1] += g.scale * 0.5 * (plus - minus)
    end
    grad
end

"Gradient from Yao's own reverse-mode autodiff."
function yao_autodiff_grad(c::Case)
    circ = yao_circuit(c)
    reg = zero_state(c.n)
    _, gparams = expect'(yao_observable(c), reg => circ)

    # `gparams` follows the order of `parameters(circ)`, which for a flat chain of
    # single-parameter rotation blocks is the order those blocks appear. Map it back onto
    # overtone-sim's parameter indices, applying the chain rule and accumulating where a
    # parameter drives several gates.
    grad = zeros(Float64, c.nparams)
    k = 0
    for g in c.gates
        if g.kind in (:RX, :RY, :RZ)
            k += 1
            g.param_index < 0 && continue
            grad[g.param_index + 1] += g.scale * gparams[k]
        end
    end
    grad
end

function main()
    path = joinpath(@__DIR__, "cases.txt")
    isfile(path) || error("$path not found. Generate it with:\n" *
        "  cargo run -p overtone-sim --example emit_oracle_cases > lab/test/cases.txt")

    cases = parse_cases(path)
    println("loaded $(length(cases)) cases from $path")

    worst_value = 0.0
    worst_ad = 0.0
    worst_shift = 0.0
    worst_internal = 0.0

    @testset "overtone-sim vs Yao.jl" begin
        for c in cases
            v = yao_value(c)
            gs = yao_shift_grad(c)
            ga = yao_autodiff_grad(c)

            dv = abs(v - c.value)
            dgs = isempty(c.grad) ? 0.0 : maximum(abs.(gs .- c.grad))
            dga = isempty(c.grad) ? 0.0 : maximum(abs.(ga .- c.grad))
            dint = isempty(gs) ? 0.0 : maximum(abs.(gs .- ga))

            worst_value = max(worst_value, dv)
            worst_shift = max(worst_shift, dgs)
            worst_ad = max(worst_ad, dga)
            worst_internal = max(worst_internal, dint)

            @test dv < TOL
            @test dgs < TOL
            @test dga < TOL
        end
    end

    @printf("worst expectation-value disagreement : %.3e\n", worst_value)
    @printf("worst gradient vs Yao parameter-shift: %.3e\n", worst_shift)
    @printf("worst gradient vs Yao autodiff       : %.3e\n", worst_ad)
    @printf("Yao shift vs Yao autodiff (self-check): %.3e\n", worst_internal)
    @printf("tolerance                            : %.3e\n", TOL)
end

main()
