using Random: seed!, randperm
using Plots, Interact, ProgressMeter, LinearAlgebra
using Random: seed!, randperm
using IJulia
theme(
    :wong;
    format=:png,
    label="",
    markerstrokewidth=0.3,
    markerstrokecolor="white",
    markersize=6,
    alpha=0.7
)

"""
Convenient way to transpose and yflip data before plotting
a heatmap
"""
function heatmap_digit(x::AbstractMatrix; kwargs...)
    return heatmap(
        x;
        showaxis=false,
        grid=false,
        transpose=true,
        yflip=true,
        color=:grays,
        aspect_ratio=1.0, kwargs...)
end
n = 10           # dimension of each sample
N = 1000         # number of samples
x = randn(n, N)  # input data
y = x' * randn(n);

linear(z::Number) = z

function dlinear(z::Number)
    return 1.0
end

function grad_loss(
        f_a::Function,
        df_a::Function,
        x::AbstractMatrix,
        y::AbstractVector,
        w::AbstractVector,
        b::Number,
        normalize::Bool=true
    )
    dw = zeros(length(w))
    db = 0.0
    loss = 0.0
    for j in 1:size(x, 2)
        error = y[j] - f_a(w' * x[:, j] +b)
        common_term = error * df_a(w' * x[:, j] + b)
        dw = dw - 2.0 * common_term .* x[:,j]
        db = db - 2.0 * common_term
        loss = loss + error^2
     end

     if normalize
        dw = dw/length(y)
        db = db/length(y)
        loss = loss/length(y)
    end
    return dw, db, loss
 end

function learn2classify_asgd(
        f_a::Function,
        df_a::Function,
        grad_loss::Function,
        x::AbstractMatrix,
        y::AbstractVector,
        mu::Number=1e-3,
        iters::Integer=500,
        batch_size::Integer=10,
        show_loss::Bool=true,
        normalize::Bool=true,
        seed::Integer=1
    )
    n, N = size(x)

    if seed == false
        b = 0.0
        w = zeros(n)
    else
        seed!(seed) # initialize random number generator
        w = randn(n)
        b = rand()
    end

    loss = zeros(iters)
    lambdak = 0
    qk = w
    pk = b
    for i in 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]

        dw, db, loss_i = grad_loss(f_a, df_a, x[:, batch_idx], y[batch_idx], w, b, normalize)
        qkp1 = w - mu * dw
        pkp1 = b - mu * db

        lambdakp1 = (1 + sqrt(1 + 4 * lambdak^2)) / 2
        gammak = (1 - lambdak) / lambdakp1

        w = (1 - gammak) * qkp1 + gammak * qk
        b = (1 - gammak) * pkp1 + gammak * pk

        qk = qkp1
        pk = pkp1
        lambdak = lambdakp1

        loss[i] = convert(Float64, loss_i[1])

        if show_loss && (rem(i, 100) == 0)
            #clear_output(true)
            loss_plot = scatter(
                [1:50:i], loss[1:50:i], yscale=:log10,
                xlabel="iteration",
                ylabel="training loss",
                title="iteration $i, loss = $loss_i"
            )
            #display(loss_plot)
            savefig(loss_plot, "/Users/Thomas/Desktop/ML-research/plot.png")
        end
    end
    return w, b, loss
end
mu = 0.00001
w_hat, b_hat, loss = learn2classify_asgd(linear, dlinear, grad_loss, x, y, mu, 1000, 10, true, true, false);
println(loss[end])
