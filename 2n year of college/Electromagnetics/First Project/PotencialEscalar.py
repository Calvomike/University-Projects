import numpy as np
import plotly.graph_objects as go

#CONSTANTES
eps0 = 8.854e-12

R = 1.0          # radio de la esfera
L = 3.0          # distancia al plano
phi0 = 1.0       # potencial de la esfera
N = 300           # número de cargas imagen

'''
qn = np.ones(N)                 # cargas imagen
zn = np.linspace(0.2, 0.8, N)   # posiciones
'''

#PARÁMETROS
xi = np.arccosh(L / R)
q0 = 4 * np.pi * eps0 * R * phi0

zn = []
qn = []

#RECURRENCIA EN LAS CARGAS Y SUS POSICIONES
for n in range(N):
    z = R * np.sinh(n * xi) / np.sinh((n + 1) * xi)
    q = q0 * np.sinh(xi) / np.sinh((n + 1) * xi)
    zn.append(z)
    qn.append(q)


#ARQUITECTURA DEL CAMPO EN LA SIMULACIÓN
nx = ny = nz = 20   # pocos puntos (si no peta)
x = np.linspace(-2.5, 2.5, nx)
y = np.linspace(-2.5, 2.5, ny)
z = np.linspace(-2.5, L-0.1, nz) #Para que se vea la perpendicularidad del campo en el plano infinito en el eje z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

#POTENCIAL ESCALAR
phi = np.zeros_like(X)

for n in range(N):
    r1 = np.sqrt(X**2 + Y**2 + (Z - zn[n])**2)
    r2 = np.sqrt(X**2 + Y**2 + (Z - (2*L - zn[n]))**2)
    phi += qn[n] * (1/r1 - 1/r2)

phi *= 1 / (4 * np.pi * eps0)

# Normalizamos solo para visualización
phi = phi / np.max(np.abs(phi))

#VISULALIZACIÓN 3D

#Figuras
# Esfera
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)

xs = R * np.outer(np.cos(u), np.sin(v))
ys = R * np.outer(np.sin(u), np.sin(v))
zs = R * np.outer(np.ones_like(u), np.cos(v))

sphere = go.Surface(
    x=xs,
    y=ys,
    z=zs,
    opacity=0.4,
    colorscale=[[0, "#E40813"], [1, "#E40813"]],
    showscale=False,
    name="Esfera"
)

# Plano conductor z = L
Xp, Yp = np.meshgrid(
    np.linspace(-3, 3, 20),
    np.linspace(-3, 3, 20)
)
Zp = L * np.ones_like(Xp)

plane = go.Surface(
    x=Xp,
    y=Yp,
    z=Zp,
    opacity=0.25,
    colorscale=[[0, "#E40813"], [1, "#E40813"]],
    showscale=False,
    name="Plano infinito"
)

#Maskeamos potencial interior de la esfera
mask = X**2 + Y**2 + Z**2 < R**2
phi[mask] = phi0

#Potencial phi
fig = go.Figure(data=[
    go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=phi.flatten(),
        isomin=-1,
        isomax=1,
        opacity=0.06,
        surface_count=200, #Numero arbitrario para que se vea bien
        colorscale="Viridis",
        name="Potencial en el espacio"
    ),
    sphere,
    plane
])

fig.update_layout(
    title="Potencial escalar φ(x,y,z) con esfera conductora y plano infinito",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="data"
    )
)

fig.show()
