import numpy as np
import matplotlib.pyplot as plt

def berakna_vinkel(A, B, C):
    BA = A - B
    BC = C - B
    dot = np.einsum('ij,ij->i', BA, BC)
    cos_vinkel = dot / (np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1))
    return np.degrees(np.arccos(np.clip(cos_vinkel, -1.0, 1.0)))

data = np.load('/Users/adambostrom/Desktop/Qualisys mätningar/Broadjump/VITpose/MotionBert/Motionbert.npy')

vanster = berakna_vinkel(data[:, 11, :], data[:, 13, :], data[:, 15, :])
hoger   = berakna_vinkel(data[:, 12, :], data[:, 14, :], data[:, 16, :])

print(vanster)
plt.plot(vanster, label='Vänster knä')
plt.plot(hoger, label='Höger knä')
plt.xlabel('Frame')
plt.ylabel('Vinkel (grader)')
plt.legend()
plt.grid(True)
plt.show()