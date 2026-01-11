import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#CONSTANTES
eps0 = 8.854e-12

R = 1.0          # radio de la esfera
L = 3.0          # distancia al plano
phi0 = 1.0       # potencial de la esfera
N = 300           # número de cargas imagen

#PARÁMETROS
xi = np.arccosh(L / R)
q0 = 4 * np.pi * eps0 * R * phi0

z = []
q = []

#RECURRENCIA EN LAS CARGAS Y SUS POSICIONES
for n in range(N):
    zn = R * np.sinh(n * xi) / np.sinh((n + 1) * xi)
    qn = q0 * np.sinh(xi) / np.sinh((n + 1) * xi)
    z.append(zn)
    q.append(qn)


charges = []

for zn, qn in zip(z, q):
    charges.append((qn, np.array([0.0, 0.0, zn])))         # dentro esfera
    charges.append((-qn, np.array([0.0, 0.0, 2*L - zn])))  # imagen del plano


#CAMPO ELECTRICO
def electric_field(r, charges):
    E = np.zeros(3)
    for q, rq in charges: #q: dentro esfera #rq: Representacion de q en el plano
        dr = r - rq
        norm = np.linalg.norm(dr)
        if norm > 1e-6:
            E += q * dr / norm**3
    return E / (4 * np.pi * eps0)



#ARQUITECTURA DEL CAMPO EN LA SIMULACIÓN
nx = ny = nz = 8   # pocos puntos (si no peta)
x = np.linspace(-2.5, 2.5, nx)
y = np.linspace(-2.5, 2.5, ny)
z = np.linspace(-2.5, L-0.1, nz) #Para que se vea la perpendicularidad del campo en el plano infinito en el eje z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)
Ez = np.zeros_like(Z)

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            r = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
            E = electric_field(r, charges)
            Ex[i,j,k] = E[0]
            Ey[i,j,k] = E[1]
            Ez[i,j,k] = E[2]


#GRAFICACIÓN

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(
    X, Y, Z,
    Ex, Ey, Ez,
    length=0.4, normalize=True
)


# Esfera
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 20)
xs = R * np.outer(np.cos(u), np.sin(v))
ys = R * np.outer(np.sin(u), np.sin(v))
zs = R * np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, color="#6708E499", alpha=0.5, label='Esfera')

# Plano conductor z = L
Xp, Yp = np.meshgrid(np.linspace(-3,3,10), np.linspace(-3,3,10))
Zp = L * np.ones_like(Xp)
ax.plot_surface(Xp, Yp, Zp, color="#6708E499", alpha=0.2, label='Plano infinito')


#Cargas imagen (dentro esfera y plano)



ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3) #Notese que se ven flexas dentro del plano para que se vea la perpendicularidad del campo en el plano infinito en el eje z
ax.set_box_aspect([1, 1, 1])

ax.set_title('Simulación del campo electrostático 3D (entrega electro.)')
plt.legend()
#plt.ion()
plt.show()
