import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib.animation as animation
import cv2  

datos_crudos = """
0.000 0.181 -8.008E-2
0.040 0.175 -9.218E-2
0.080 0.171 -0.101
0.120 0.169 -0.115
0.160 0.165 -0.124
0.200 0.162 -0.129
0.240 0.158 -0.133
0.280 0.160 -0.135
0.320 0.162 -0.135
0.360 0.163 -0.133
0.400 0.165 -0.131
0.440 0.167 -0.128
0.480 0.172 -0.122
0.520 0.176 -0.112
0.560 0.183 -9.679E-2
0.600 0.185 -8.757E-2
0.640 0.194 -6.625E-2
0.680 0.198 -4.839E-2
0.720 0.198 -2.420E-2
0.760 0.206 1.728E-3
0.800 0.208 2.420E-2
0.840 0.207 3.745E-2
0.880 0.206 5.127E-2
0.920 0.204 5.415E-2
0.960 0.204 5.358E-2
1.000 0.203 5.070E-2
1.040 0.202 4.839E-2
1.080 0.201 4.436E-2
1.120 0.200 4.148E-2
1.160 0.198 3.572E-2
1.200 0.198 2.708E-2
1.240 0.196 1.728E-2
1.280 0.196 6.913E-3
1.320 0.197 -4.033E-3
1.360 0.195 -1.555E-2
1.400 0.192 -2.765E-2
1.440 0.189 -4.148E-2
1.480 0.188 -5.703E-2
1.520 0.183 -7.144E-2
1.560 0.181 -8.584E-2
1.600 0.180 -9.679E-2
1.640 0.176 -0.111
1.680 0.172 -0.122
1.720 0.166 -0.130
1.760 0.161 -0.139
1.800 0.158 -0.147
1.840 0.156 -0.152
1.880 0.155 -0.158
1.920 0.153 -0.161
1.960 0.152 -0.158
2.000 0.157 -0.153
2.040 0.158 -0.145
2.080 0.165 -0.135
2.120 0.169 -0.113
2.160 0.177 -0.104
2.200 0.186 -8.065E-2
2.240 0.190 -5.358E-2
2.280 0.201 -1.613E-2
2.320 0.203 1.037E-2
2.360 0.202 3.284E-2
2.400 0.200 4.954E-2
2.440 0.194 6.625E-2
2.480 0.188 8.008E-2
2.520 0.184 8.930E-2
2.560 0.180 9.736E-2
2.600 0.177 9.621E-2
2.640 0.172 8.872E-2
2.680 0.173 7.201E-2
2.720 0.179 5.531E-2
2.760 0.180 2.592E-2
2.800 0.182 -2.776E-17
2.840 0.184 -2.823E-2
2.880 0.184 -5.012E-2
2.920 0.184 -7.144E-2
2.960 0.182 -9.102E-2
3.000 0.179 -0.108
3.040 0.176 -0.123
3.080 0.175 -0.133
3.120 0.175 -0.142
3.160 0.176 -0.147
3.200 0.176 -0.150
3.240 0.180 -0.146
3.280 0.184 -0.139
3.320 0.190 -0.134
3.360 0.191 -0.128
3.400 0.192 -0.119
3.440 0.196 -0.106
3.480 0.198 -9.102E-2
3.520 0.200 -7.720E-2
3.560 0.206 -6.164E-2
3.600 0.209 -3.975E-2
3.640 0.210 -9.218E-3
3.680 0.210 1.555E-2
3.720 0.208 3.629E-2
3.760 0.207 5.646E-2
3.800 0.202 7.605E-2
3.840 0.199 9.045E-2
3.880 0.198 0.105
3.920 0.190 0.120
3.960 0.183 0.135
4.000 0.179 0.147
4.040 0.170 0.156
4.080 0.169 0.153
4.120 0.169 0.148
4.160 0.171 0.136
4.200 0.170 0.116
4.240 0.171 9.563E-2
4.280 0.172 7.317E-2
4.320 0.176 5.070E-2
4.360 0.176 2.650E-2
4.400 0.176 -5.761E-4
4.440 0.173 -3.111E-2
4.480 0.166 -5.300E-2
4.520 0.165 -7.201E-2
4.560 0.171 -9.160E-2
4.600 0.171 -0.105
4.640 0.169 -0.118
4.680 0.171 -0.123
4.720 0.177 -0.127
4.760 0.180 -0.127
4.800 0.183 -0.120
4.840 0.182 -0.109
4.880 0.187 -9.851E-2
4.920 0.191 -8.526E-2
4.960 0.195 -6.913E-2
5.000 0.202 -5.012E-2
5.040 0.202 -3.284E-2
5.080 0.203 -1.152E-2
5.120 0.200 1.786E-2
5.160 0.207 3.745E-2
5.200 0.207 5.761E-2
5.240 0.203 7.432E-2
5.280 0.202 9.160E-2
5.320 0.199 0.106
5.360 0.191 0.117
5.400 0.184 0.130
5.440 0.180 0.142
"""

data = np.loadtxt(io.StringIO(datos_crudos))
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

def resolver_spline_cubico(t_pts, y_pts):
    N = len(t_pts)
    h = np.diff(t_pts)
    A = np.zeros((N-2, N-2))
    B = np.zeros(N-2)
    for i in range(1, N-1):
        r = i - 1
        if r > 0:
            A[r, r-1] = h[i-1]
        A[r, r] = 2.0 * (h[i-1] + h[i])
        if r < N-3:
            A[r, r+1] = h[i]
        B[r] = 3.0 * ((y_pts[i+1] - y_pts[i])/h[i] - (y_pts[i] - y_pts[i-1])/h[i-1])
    c_mid = np.linalg.solve(A, B)
    c = np.zeros(N)
    c[1:N-1] = c_mid
    a = y_pts[:-1]
    b = np.zeros(N-1)
    d = np.zeros(N-1)
    for i in range(N-1):
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i])
        b[i] = (y_pts[i+1] - y_pts[i])/h[i] - h[i] * (c[i+1] + 2.0 * c[i]) / 3.0
    return a, b, c[:-1], d

def evaluar_spline(t_query, a, b, c, d, t_nodes):
    indices = np.searchsorted(t_nodes, t_query) - 1
    indices = np.clip(indices, 0, len(t_nodes) - 2)
    dt = t_query - t_nodes[indices]
    return a[indices] + b[indices] * dt + c[indices] * (dt**2) + d[indices] * (dt**3)

ax_spline, bx_spline, cx_spline, dx_spline = resolver_spline_cubico(t, x)
ay_spline, by_spline, cy_spline, dy_spline = resolver_spline_cubico(t, y)

t_alta_res = np.linspace(t[0], t[-1], 1000)
x_ruta = evaluar_spline(t_alta_res, ax_spline, bx_spline, cx_spline, dx_spline, t)
y_ruta = evaluar_spline(t_alta_res, ay_spline, by_spline, cy_spline, dy_spline, t)

fig, (ax_vid, ax_anim) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Comparativa Cinematica: Cernicalo", fontsize=16, fontweight='bold')

cap = cv2.VideoCapture('video-vista.mp4')

ret, primer_frame = cap.read()
if ret:
    primer_frame = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2RGB)
    video_display = ax_vid.imshow(primer_frame)
    
ax_vid.set_title("Video Real (Vuelo Estacionario)", fontsize=14)
ax_vid.axis('off')

ax_anim.plot(x, y, 'o', color='gray', alpha=0.3, label='Puntos de Tracker')
ax_anim.plot(x_ruta, y_ruta, '-', color='#50C878', linewidth=2, alpha=0.6, label='Ruta Spline Cubico')
punto_viajero, = ax_anim.plot([], [], 'o', color='#0B6623', markersize=10, label='Punta del Ala')

ax_anim.set_title("Modelo Matematico (Splines)", fontsize=14)
ax_anim.set_xlabel("X (m)")
ax_anim.set_ylabel("Y (m)")
ax_anim.grid(True, linestyle='--', alpha=0.7)
ax_anim.legend(loc='upper right')
ax_anim.axis('equal')

frames_totales = len(t)

def inicializar_animacion():
    punto_viajero.set_data([], [])
    return video_display, punto_viajero

def actualizar_animacion(frame):
    ret, frame_vid = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_vid = cap.read()
    if ret:
        frame_vid = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
        video_display.set_data(frame_vid)
    tiempo_actual = t[frame]
    px = evaluar_spline(tiempo_actual, ax_spline, bx_spline, cx_spline, dx_spline, t)
    py = evaluar_spline(tiempo_actual, ay_spline, by_spline, cy_spline, dy_spline, t)
    punto_viajero.set_data([px], [py])
    return video_display, punto_viajero

ani = animation.FuncAnimation(fig, 
                              actualizar_animacion, 
                              frames=frames_totales, 
                              init_func=inicializar_animacion, 
                              blit=True, 
                              interval=40, 
                              repeat=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

cap.release()