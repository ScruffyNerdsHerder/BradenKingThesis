using Plots
using JLD2
is = 0:96
Phis = 1/16*pi.*is
rs = 6.5.*(104 .- is)/104

x = rs.*cos.(Phis)
y = rs.*sin.(Phis)

x2 = -rs.*cos.(Phis)
y2 = -rs.*sin.(Phis)

X = vcat(hcat(y,x), hcat(y2,x2))

plot(X[:,1], X[:,2], seriestype=:scatter, label="", aspect_ratio=1, title="Spiral Data", xlabel="X", ylabel="Y")

@save("SpiralData.jld2", X)

