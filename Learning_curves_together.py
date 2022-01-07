import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
import matplotlib.pylab as plt


'''First, store all the results as arrays. Then use the display the them on a graph'''
snn = { 10: 0.714828897338403, 11: 0.714828897338403, 12: 0.714828897338403, 13: 0.7203065134099617, 14: 0.7286821705426356, 15: 0.714828897338403, 16: 0.7258687258687259, 17: 0.714828897338403, 18: 0.714828897338403, 19: 0.714828897338403, 20: 0.7203065134099617, 21: 0.7286821705426356, 22: 0.714828897338403, 23: 0.7286821705426356, 24: 0.714828897338403, 25: 0.7286821705426356, 26: 0.714828897338403, 27: 0.7401574803149606, 28: 0.7580645161290321, 29: 0.7704918032786886, 30: 0.8068669527896996, 31: 0.8392857142857142, 32: 0.8744186046511627, 33: 0.8867924528301886, 34: 0.9082125603864734, 35: 0.8952380952380952, 36: 0.9082125603864734, 37: 0.9215686274509803, 38: 0.8995215311004785, 39: 0.8952380952380952, 40: 0.8952380952380952, 41: 0.9038461538461539, 42: 0.8952380952380952, 43: 0.9215686274509803, 44: 0.8995215311004785, 45: 0.9126213592233009, 46: 0.8867924528301886, 47: 0.8952380952380952, 48: 0.9126213592233009, 49: 0.9038461538461539, 50: 0.8995215311004785, 51: 0.9126213592233009, 52: 0.9082125603864734, 53: 0.9038461538461539, 54: 0.9082125603864734, 55: 0.8909952606635071, 56: 0.8995215311004785, 57: 0.8995215311004785, 58: 0.9082125603864734, 59: 0.9170731707317074, 60: 0.9261083743842364, 61: 0.9261083743842364, 62: 0.9543147208121827, 63: 0.9494949494949495, 64: 0.9447236180904524, 65: 0.9494949494949495, 66: 0.9400000000000001, 67: 0.9261083743842364, 68: 0.9447236180904524, 69: 0.9400000000000001, 70: 0.9400000000000001, 71: 0.9494949494949495, 72: 0.9447236180904524, 73: 0.9400000000000001, 74: 0.9306930693069307, 75: 0.9306930693069307, 76: 0.9353233830845772, 77: 0.9306930693069307, 78: 0.9353233830845772, 79: 0.9400000000000001, 80: 0.9400000000000001, 81: 0.9591836734693878, 82: 0.9494949494949495, 83: 0.9690721649484536, 84: 0.9543147208121827, 85: 0.9738219895287958, 86: 0.9740932642487047, 87: 0.9791666666666666, 88: 0.9637305699481866, 89: 0.9690721649484536, 90: 0.9789473684210526, 91: 0.9637305699481866, 92: 0.9740932642487047, 93: 0.9591836734693878, 94: 0.9641025641025642, 95: 0.9591836734693878, 96: 0.9543147208121827, 97: 0.9591836734693878, 98: 0.9494949494949495, 99: 0.9543147208121827, 100: 0.9690721649484536, 101: 0.9641025641025642, 102: 0.9494949494949495, 103: 0.9543147208121827, 104: 0.9543147208121827, 105: 0.9641025641025642, 106: 0.9591836734693878, 107: 0.9591836734693878, 108: 0.9543147208121827, 109: 0.9543147208121827, 110: 0.9791666666666666, 111: 0.9740932642487047, 112: 0.9791666666666666, 113: 0.9740932642487047, 114: 0.9842931937172774, 115: 0.9791666666666666, 116: 0.9842931937172774, 117: 0.9791666666666666, 118: 0.9791666666666666, 119: 0.9791666666666666, 120: 0.9842931937172774, 121: 0.9791666666666666, 122: 0.9842931937172774, 123: 0.968421052631579, 124: 0.9842931937172774, 125: 0.9842931937172774, 126: 0.9842931937172774, 127: 0.9791666666666666, 128: 0.9789473684210526, 129: 0.9740932642487047, 130: 0.9791666666666666, 131: 0.9791666666666666, 132: 0.9740932642487047, 133: 0.9690721649484536, 134: 0.9740932642487047, 135: 0.9641025641025642, 136: 0.9791666666666666, 137: 0.9740932642487047, 138: 0.9791666666666666, 139: 0.9791666666666666, 140: 0.9740932642487047, 141: 0.9690721649484536, 142: 0.9641025641025642, 143: 0.9591836734693878, 144: 0.9591836734693878, 145: 0.9494949494949495, 146: 0.9494949494949495, 147: 0.9543147208121827, 148: 0.9400000000000001, 149: 0.9447236180904524, 150: 0.9494949494949495, 151: 0.9447236180904524, 152: 0.9591836734693878, 153: 0.9641025641025642, 154: 0.9740932642487047, 155: 0.9494949494949495, 156: 0.9591836734693878, 157: 0.9591836734693878, 158: 0.9494949494949495, 159: 0.9353233830845772, 160: 0.9353233830845772, 161: 0.9353233830845772, 162: 0.9400000000000001, 163: 0.9447236180904524, 164: 0.9400000000000001, 165: 0.9400000000000001, 166: 0.9353233830845772, 167: 0.9353233830845772, 168: 0.9400000000000001, 169: 0.9261083743842364, 170: 0.9306930693069307, 171: 0.9261083743842364, 172: 0.9400000000000001, 173: 0.9400000000000001, 174: 0.9447236180904524, 175: 0.9641025641025642, 176: 0.9494949494949495, 177: 0.9447236180904524, 178: 0.9447236180904524, 179: 0.9447236180904524, 180: 0.9353233830845772, 181: 0.9447236180904524, 182: 0.9494949494949495, 183: 0.9447236180904524, 184: 0.9494949494949495, 185: 0.9494949494949495, 186: 0.9447236180904524, 187: 0.9494949494949495, 188: 0.9591836734693878, 189: 0.9400000000000001, 190: 0.9543147208121827, 191: 0.9591836734693878, 192: 0.9494949494949495, 193: 0.9447236180904524, 194: 0.9447236180904524, 195: 0.9447236180904524, 196: 0.9400000000000001, 197: 0.9494949494949495, 198: 0.9543147208121827, 199: 0.9494949494949495, 200: 0.9447236180904524, 201: 0.9447236180904524, 202: 0.9494949494949495, 203: 0.9494949494949495, 204: 0.9447236180904524, 205: 0.9447236180904524, 206: 0.9447236180904524, 207: 0.9447236180904524, 208: 0.9543147208121827, 209: 0.9543147208121827, 210: 0.9447236180904524, 211: 0.9591836734693878, 212: 0.9494949494949495, 213: 0.9400000000000001, 214: 0.9447236180904524, 215: 0.9400000000000001, 216: 0.9543147208121827, 217: 0.9353233830845772, 218: 0.9447236180904524, 219: 0.9353233830845772, 220: 0.9353233830845772, 221: 0.9447236180904524, 222: 0.9447236180904524, 223: 0.9400000000000001, 224: 0.9494949494949495, 225: 0.9353233830845772, 226: 0.9400000000000001, 227: 0.9400000000000001, 228: 0.9400000000000001, 229: 0.9400000000000001, 230: 0.9494949494949495, 231: 0.9353233830845772, 232: 0.9353233830845772, 233: 0.9543147208121827, 234: 0.9261083743842364, 235: 0.9261083743842364, 236: 0.9400000000000001, 237: 0.9353233830845772, 238: 0.9400000000000001, 239: 0.9261083743842364, 240: 0.9306930693069307, 241: 0.9306930693069307, 242: 0.9353233830845772, 243: 0.9400000000000001, 244: 0.9306930693069307, 245: 0.9353233830845772, 246: 0.9306930693069307, 247: 0.9353233830845772, 248: 0.9306930693069307, 249: 0.9353233830845772, 250: 0.9353233830845772, 251: 0.9261083743842364, 252: 0.9353233830845772, 253: 0.9447236180904524, 254: 0.9261083743842364, 255: 0.9353233830845772, 256: 0.9353233830845772, 257: 0.9261083743842364, 258: 0.9353233830845772, 259: 0.9543147208121827, 260: 0.9494949494949495, 261: 0.9543147208121827, 262: 0.9447236180904524, 263: 0.9591836734693878, 264: 0.9591836734693878, 265: 0.9543147208121827, 266: 0.9543147208121827, 267: 0.9591836734693878, 268: 0.9447236180904524, 269: 0.9543147208121827, 270: 0.9494949494949495, 271: 0.9591836734693878, 272: 0.9543147208121827, 273: 0.9543147208121827, 274: 0.9591836734693878, 275: 0.9591836734693878, 276: 0.9494949494949495, 277: 0.9494949494949495, 278: 0.9543147208121827, 279: 0.9591836734693878, 280: 0.9543147208121827, 281: 0.9543147208121827, 282: 0.9591836734693878, 283: 0.9591836734693878, 284: 0.9591836734693878, 285: 0.9591836734693878, 286: 0.9494949494949495, 287: 0.9543147208121827, 288: 0.9543147208121827, 289: 0.9494949494949495, 290: 0.9543147208121827, 291: 0.9543147208121827, 292: 0.9543147208121827, 293: 0.9543147208121827, 294: 0.9591836734693878, 295: 0.9353233830845772, 296: 0.9494949494949495, 297: 0.9447236180904524, 298: 0.9494949494949495, 299: 0.9494949494949495, 300: 0.9447236180904524, 301: 0.9494949494949495, 302: 0.9447236180904524, 303: 0.9447236180904524, 304: 0.9494949494949495, 305: 0.9447236180904524, 306: 0.9494949494949495, 307: 0.9400000000000001, 308: 0.9447236180904524, 309: 0.9591836734693878, 310: 0.9591836734693878, 311: 0.9543147208121827, 312: 0.9591836734693878, 313: 0.9543147208121827, 314: 0.9494949494949495, 315: 0.9543147208121827, 316: 0.9641025641025642, 317: 0.9591836734693878, 318: 0.9543147208121827, 319: 0.9591836734693878, 320: 0.9591836734693878, 321: 0.9641025641025642, 322: 0.9591836734693878, 323: 0.9591836734693878, 324: 0.9591836734693878, 325: 0.9591836734693878, 326: 0.9591836734693878, 327: 0.9641025641025642, 328: 0.9591836734693878, 329: 0.9494949494949495, 330: 0.9641025641025642, 331: 0.9543147208121827, 332: 0.9591836734693878, 333: 0.9543147208121827, 334: 0.9591836734693878, 335: 0.9591836734693878, 336: 0.9690721649484536, 337: 0.9591836734693878, 338: 0.9543147208121827, 339: 0.9543147208121827, 340: 0.9543147208121827, 341: 0.9690721649484536, 342: 0.9641025641025642, 343: 0.9690721649484536, 344: 0.9591836734693878, 345: 0.9494949494949495, 346: 0.9591836734693878, 347: 0.9641025641025642, 348: 0.9591836734693878, 349: 0.9543147208121827, 350: 0.9591836734693878, 351: 0.9591836734693878, 352: 0.9641025641025642, 353: 0.9641025641025642, 354: 0.9591836734693878, 355: 0.9591836734693878, 356: 0.9690721649484536, 357: 0.9591836734693878, 358: 0.9494949494949495, 359: 0.9494949494949495, 360: 0.9641025641025642, 361: 0.9447236180904524, 362: 0.9494949494949495, 363: 0.9591836734693878, 364: 0.9591836734693878, 365: 0.9494949494949495, 366: 0.9494949494949495, 367: 0.9591836734693878, 368: 0.9494949494949495, 369: 0.9591836734693878, 370: 0.9543147208121827, 371: 0.9591836734693878, 372: 0.9641025641025642, 373: 0.9591836734693878, 374: 0.9543147208121827, 375: 0.9591836734693878, 376: 0.9494949494949495, 377: 0.9543147208121827, 378: 0.9494949494949495, 379: 0.9543147208121827, 380: 0.9591836734693878, 381: 0.9591836734693878, 382: 0.9641025641025642, 383: 0.9494949494949495, 384: 0.9447236180904524, 385: 0.9400000000000001, 386: 0.9400000000000001, 387: 0.9447236180904524, 388: 0.9400000000000001, 389: 0.9400000000000001, 390: 0.9400000000000001, 391: 0.9447236180904524, 392: 0.9353233830845772, 393: 0.9400000000000001, 394: 0.9447236180904524, 395: 0.9400000000000001, 396: 0.9447236180904524, 397: 0.9447236180904524, 398: 0.9494949494949495, 399: 0.9543147208121827}
dnn = [
 (5, 0.7372549019607844),
 (10, 0.7782426778242678),
 (15, 0.7372549019607844),
 (20, 0.7230769230769232),
 (25, 0.714828897338403),
 (30, 0.8209606986899564),
 (35, 0.9157894736842105),
 (40, 0.8611111111111112),
 (45, 0.9183673469387754),
 (50, 0.8479262672811059),
 (55, 0.9387755102040817),
 (60, 0.9518716577540107),
 (65, 0.9479166666666666),
 (70, 0.9565217391304347),
 (75, 0.9306930693069307),
 (80, 0.9468085106382979),
 (85, 0.9393939393939394),
 (90, 0.9381443298969071),
 (95, 0.9574468085106385),
 (100, 0.9393939393939394),
 (105, 0.9690721649484536),
 (110, 0.972972972972973),
 (115, 0.972972972972973),
 (120, 0.968421052631579),
 (125, 0.9735449735449735),
 (130, 0.9680851063829787),
 (135, 0.963350785340314),
 (140, 0.9625668449197862),
 (145, 0.968421052631579),
 (150, 0.9583333333333333),
 (155, 0.9735449735449735),
 (160, 0.9583333333333333),
 (165, 0.9680851063829787),
 (170, 0.96875),
 (175, 0.9735449735449735),
 (180, 0.9591836734693878),
 (185, 0.96875),
 (190, 0.9637305699481866),
 (195, 0.9732620320855615),
 (200, 0.9543147208121827),
 (205, 0.963350785340314),
 (210, 0.96875),
 (215, 0.9740932642487047),
 (220, 0.9489795918367346),
 (225, 0.9494949494949495),
 (230, 0.967741935483871),
 (235, 0.968421052631579),
 (240, 0.968421052631579),
 (245, 0.9690721649484536),
 (250, 0.9791666666666666),
 (255, 0.96875),
 (260, 0.968421052631579),
 (265, 0.9538461538461538),
 (270, 0.9637305699481866),
 (275, 0.9690721649484536),
 (280, 0.9587628865979382),
 (285, 0.9690721649484536),
 (290, 0.9637305699481866),
 (295, 0.96875),
 (300, 0.9690721649484536),
 (305, 0.9543147208121827),
 (310, 0.96875),
 (315, 0.9641025641025642),
 (320, 0.9637305699481866),
 (325, 0.9637305699481866),
 (330, 0.968421052631579),
 (335, 0.9690721649484536),
 (340, 0.96875),
 (345, 0.96875),
 (350, 0.9591836734693878),
 (355, 0.9543147208121827),
 (360, 0.963350785340314),
 (365, 0.9690721649484536),
 (370, 0.9543147208121827),
 (375, 0.9587628865979382),
 (380, 0.9587628865979382),
 (385, 0.96875),
 (390, 0.9494949494949495),
 (395, 0.9591836734693878),
 (400, 0.9591836734693878)]
#Bayes and DT has the first index of 5
bayes = [0.714828897338403, 0.714828897338403, 0.714828897338403, 0.714828897338403, 0.714828897338403, 0.714828897338403, 0.714828897338403, 0.714828897338403, 0.8744186046511627, 0.8785046728971964, 0.8744186046511627, 0.8826291079812207, 0.8430493273542601, 0.8545454545454546, 0.8584474885844748, 0.8584474885844748, 0.8995215311004785, 0.9082125603864734, 0.9082125603864734, 0.9126213592233009, 0.9126213592233009, 0.9126213592233009, 0.916256157635468, 0.9207920792079208, 0.9207920792079208, 0.9207920792079208, 0.916256157635468, 0.9207920792079208, 0.9207920792079208, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9207920792079208, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9292929292929293, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.9253731343283582, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.9246231155778896, 0.9246231155778896, 0.9246231155778896, 0.9246231155778896, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9246231155778896, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.9346733668341708, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9346733668341708, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9253731343283582, 0.93, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.9346733668341708, 0.9346733668341708, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]
dt = [0.7520661157024794, 0.8325358851674641, 0.8380952380952381, 0.8744186046511627, 0.8378378378378378, 0.7727272727272727, 0.8544600938967135, 0.8544600938967135, 0.8504672897196263, 0.8416289592760181, 0.8465116279069769, 0.8266666666666667, 0.810344827586207, 0.8703703703703703, 0.8703703703703703, 0.8465116279069769, 0.8544600938967135, 0.8504672897196263, 0.8695652173913044, 0.8703703703703703, 0.8703703703703703, 0.8430493273542601, 0.852791878172589, 0.8944723618090452, 0.8629441624365481, 0.9137055837563451, 0.9137055837563451, 0.8629441624365481, 0.8629441624365481, 0.9137055837563451, 0.9137055837563451, 0.8629441624365481, 0.8629441624365481, 0.8629441624365481, 0.8542713567839195, 0.8629441624365481, 0.9, 0.9, 0.8542713567839195, 0.9, 0.9, 0.9, 0.9, 0.89, 0.8629441624365481, 0.9137055837563451, 0.8866995073891627, 0.8866995073891627, 0.8866995073891627, 0.8866995073891627, 0.9137055837563451, 0.9137055837563451, 0.8866995073891627, 0.9137055837563451, 0.9137055837563451, 0.8866995073891627, 0.9137055837563451, 0.9137055837563451, 0.8866995073891627, 0.9137055837563451, 0.9137055837563451, 0.9137055837563451, 0.9137055837563451, 0.9137055837563451, 0.8673469387755102, 0.9081632653061226, 0.8497409326424871, 0.8969072164948454, 0.9081632653061226, 0.9025641025641027, 0.8795811518324608, 0.9081632653061226, 0.9285714285714286, 0.934010152284264, 0.9285714285714286, 0.9015544041450778, 0.9128205128205128, 0.90625, 0.8947368421052632, 0.8900523560209425, 0.9072164948453608, 0.9072164948453608, 0.9119170984455959, 0.90625, 0.9072164948453608, 0.9015544041450778, 0.8958333333333333, 0.90625, 0.9072164948453608, 0.9222797927461138, 0.9025641025641027, 0.8877005347593583, 0.9222797927461138, 0.8877005347593583, 0.9222797927461138, 0.9222797927461138, 0.9222797927461138, 0.9222797927461138, 0.9166666666666666, 0.9166666666666666, 0.9222797927461138, 0.9137055837563451, 0.9222797927461138, 0.9025641025641027, 0.9222797927461138, 0.9166666666666666, 0.9175257731958764, 0.9222797927461138, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9166666666666666, 0.9119170984455959, 0.9166666666666666, 0.9222797927461138, 0.9166666666666666, 0.9183673469387754, 0.9222797927461138, 0.9222797927461138, 0.9175257731958764, 0.9222797927461138, 0.9222797927461138, 0.9119170984455959, 0.9222797927461138, 0.9222797927461138, 0.9072164948453608, 0.9128205128205128, 0.9128205128205128, 0.9072164948453608, 0.9072164948453608, 0.9166666666666666, 0.9166666666666666, 0.9090909090909091, 0.9090909090909091, 0.9035532994923857, 0.9090909090909091, 0.9090909090909091, 0.9035532994923857, 0.9128205128205128, 0.9166666666666666, 0.9128205128205128, 0.9015544041450778, 0.9072164948453608, 0.9072164948453608, 0.9090909090909091, 0.8979591836734694, 0.9072164948453608, 0.9035532994923857, 0.9183673469387754, 0.9183673469387754, 0.9072164948453608, 0.9128205128205128, 0.9035532994923857, 0.9090909090909091, 0.9128205128205128, 0.9128205128205128, 0.9072164948453608, 0.9183673469387754, 0.9090909090909091, 0.9128205128205128, 0.9072164948453608, 0.9072164948453608, 0.9222797927461138, 0.9166666666666666, 0.9090909090909091, 0.9222797927461138, 0.9090909090909091, 0.9128205128205128, 0.9035532994923857, 0.9081632653061226, 0.9081632653061226, 0.9128205128205128, 0.9128205128205128, 0.90625, 0.9109947643979057, 0.9183673469387754, 0.9072164948453608, 0.9081632653061226, 0.90625, 0.90625, 0.9128205128205128, 0.9025641025641027, 0.9025641025641027, 0.9222797927461138, 0.9222797927461138, 0.9128205128205128, 0.9214659685863875, 0.9214659685863875, 0.9081632653061226, 0.9119170984455959, 0.90625, 0.9081632653061226, 0.9214659685863875, 0.9045226130653266, 0.9025641025641027, 0.9081632653061226, 0.8934010152284263, 0.9157894736842105, 0.898989898989899, 0.898989898989899, 0.9157894736842105, 0.9119170984455959, 0.9081632653061226, 0.9175257731958764, 0.9025641025641027, 0.9214659685863875, 0.9081632653061226, 0.9025641025641027, 0.9045226130653266, 0.898989898989899, 0.9081632653061226, 0.911764705882353, 0.9090909090909091, 0.9108910891089108, 0.8944723618090452, 0.9082125603864734, 0.8975609756097561, 0.9035532994923857, 0.9246231155778896, 0.9199999999999999, 0.9154228855721392, 0.9054726368159204, 0.9090909090909091, 0.9154228855721392, 0.9029126213592233, 0.9170731707317074, 0.9333333333333335, 0.93, 0.9207920792079208, 0.9081632653061226, 0.9045226130653266, 0.9214659685863875, 0.8944723618090452, 0.9387755102040817, 0.9175257731958764, 0.9045226130653266, 0.9215686274509803, 0.9175257731958764, 0.9081632653061226, 0.8979591836734694, 0.9175257731958764, 0.9191919191919191, 0.9270833333333334, 0.9137055837563451, 0.9393939393939394, 0.9230769230769231, 0.9215686274509803, 0.9081632653061226, 0.9230769230769231, 0.9035532994923857, 0.9045226130653266, 0.9154228855721392, 0.9261083743842364, 0.9214659685863875, 0.9175257731958764, 0.9035532994923857, 0.9128205128205128, 0.91, 0.9081632653061226, 0.9175257731958764, 0.9045226130653266, 0.9137055837563451, 0.9045226130653266, 0.9175257731958764, 0.9081632653061226, 0.9025641025641027, 0.9137055837563451, 0.9045226130653266, 0.9025641025641027, 0.9045226130653266, 0.9025641025641027, 0.9081632653061226, 0.9137055837563451, 0.9045226130653266, 0.9081632653061226, 0.9045226130653266, 0.9045226130653266, 0.898989898989899, 0.9081632653061226, 0.9081632653061226, 0.9081632653061226, 0.8768472906403939, 0.8682926829268293, 0.8743718592964825, 0.883248730964467, 0.8844221105527638, 0.8944723618090452, 0.8944723618090452, 0.8669950738916257, 0.9, 0.8888888888888888, 0.8656716417910447, 0.8985507246376812, 0.9064039408866995, 0.9215686274509803, 0.8975609756097561, 0.9073170731707317, 0.8932038834951456, 0.8942307692307693, 0.898989898989899, 0.916256157635468, 0.9073170731707317, 0.9035532994923857, 0.8756218905472636, 0.8899521531100479, 0.9207920792079208, 0.9306930693069307, 0.9353233830845772, 0.9119170984455959, 0.9253731343283582, 0.9215686274509803, 0.9261083743842364, 0.9261083743842364, 0.93, 0.9353233830845772, 0.9353233830845772, 0.9346733668341708, 0.9253731343283582, 0.9207920792079208, 0.9253731343283582, 0.9261083743842364, 0.9261083743842364, 0.9261083743842364, 0.9353233830845772, 0.9253731343283582, 0.9253731343283582, 0.9261083743842364, 0.9261083743842364, 0.9261083743842364, 0.9353233830845772, 0.9346733668341708, 0.9261083743842364, 0.9207920792079208, 0.93, 0.93, 0.9207920792079208, 0.9346733668341708, 0.9145728643216081, 0.9306930693069307, 0.9090909090909091, 0.9306930693069307, 0.9145728643216081, 0.9306930693069307, 0.9393939393939394, 0.9108910891089108, 0.9400000000000001, 0.9154228855721392, 0.9400000000000001, 0.9400000000000001, 0.9253731343283582, 0.9191919191919191, 0.9108910891089108, 0.9199999999999999, 0.9353233830845772, 0.93, 0.9346733668341708, 0.9261083743842364, 0.9353233830845772, 0.93, 0.9261083743842364, 0.9400000000000001, 0.9353233830845772, 0.9353233830845772, 0.9253731343283582, 0.93, 0.9253731343283582, 0.9253731343283582, 0.93, 0.93, 0.9261083743842364, 0.9261083743842364, 0.9353233830845772, 0.9353233830845772, 0.9261083743842364, 0.93, 0.9253731343283582, 0.9353233830845772, 0.9400000000000001, 0.9353233830845772, 0.9353233830845772, 0.9353233830845772, 0.9400000000000001, 0.9353233830845772, 0.9253731343283582, 0.93, 0.9261083743842364, 0.9261083743842364, 0.9306930693069307, 0.9261083743842364, 0.9353233830845772, 0.9215686274509803, 0.9306930693069307, 0.9353233830845772, 0.9306930693069307, 0.9261083743842364, 0.9447236180904524]
snn_gc = [ (5, 0.714828897338403),
 (10, 0.714828897338403),
 (15, 0.7175572519083969),
 (20, 0.714828897338403),
 (25, 0.7203065134099617),
 (30, 0.7768595041322314),
 (35, 0.8584474885844748),
 (40, 0.8867924528301886),
 (45, 0.8826291079812207),
 (50, 0.9073170731707317),
 (55, 0.8867924528301886),
 (60, 0.8942307692307693),
 (65, 0.9261083743842364),
 (70, 0.9207920792079208),
 (75, 0.9073170731707317),
 (80, 0.934010152284264),
 (85, 0.934010152284264),
 (90, 0.9387755102040817),
 (95, 0.9489795918367346),
 (100, 0.9393939393939394),
 (105, 0.934010152284264),
 (110, 0.9538461538461538),
 (115, 0.9484536082474226),
 (120, 0.962962962962963),
 (125, 0.9381443298969071),
 (130, 0.9680851063829787),
 (135, 0.9489795918367346),
 (140, 0.962962962962963),
 (145, 0.9441624365482233),
 (150, 0.9435897435897436),
 (155, 0.9393939393939394),
 (160, 0.9393939393939394),
 (165, 0.9441624365482233),
 (170, 0.9587628865979382),
 (175, 0.9346733668341708),
 (180, 0.9489795918367346),
 (185, 0.9489795918367346),
 (190, 0.9441624365482233),
 (195, 0.9441624365482233),
 (200, 0.9441624365482233),
 (205, 0.9393939393939394),
 (210, 0.9587628865979382),
 (215, 0.9393939393939394),
 (220, 0.9253731343283582),
 (225, 0.9489795918367346),
 (230, 0.9441624365482233),
 (235, 0.93),
 (240, 0.93),
 (245, 0.9393939393939394),
 (250, 0.93),
 (255, 0.9393939393939394),
 (260, 0.9538461538461538),
 (265, 0.9253731343283582),
 (270, 0.9346733668341708),
 (275, 0.9253731343283582),
 (280, 0.9393939393939394),
 (285, 0.9253731343283582),
 (290, 0.9441624365482233),
 (295, 0.9346733668341708),
 (300, 0.9538461538461538),
 (305, 0.9538461538461538),
 (310, 0.9587628865979382),
 (315, 0.9207920792079208),
 (320, 0.9538461538461538),
 (325, 0.9346733668341708),
 (330, 0.9441624365482233),
 (335, 0.9073170731707317),
 (340, 0.93),
 (345, 0.9207920792079208),
 (350, 0.9207920792079208),
 (355, 0.9393939393939394),
 (360, 0.9346733668341708),
 (365, 0.9587628865979382),
 (370, 0.9637305699481866),
 (375, 0.916256157635468),
 (380, 0.9393939393939394),
 (385, 0.9393939393939394),
 (390, 0.9393939393939394),
 (395, 0.9637305699481866),
 (400, 0.9441624365482233)]

bayes_indexes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400]

def display_graph(title):
    plt.yticks(np.arange(0.5, 1, 0.05))
    plt.ylabel('F1 Score')
    plt.xlabel('Number of training data')
    plt.grid(True)
    plt.title(title)
    plt.show()

#snn
plot_list = sorted(snn.items())
x_cord, y_cord = zip(*plot_list)
plt.plot(x_cord, y_cord, label = 'SNN', color = 'tab:blue')

#dnn
plot_list = dnn
x_cord, y_cord = zip(*plot_list)
plt.plot(x_cord, y_cord, label = 'DNN', color = 'tab:orange')

#bayes
x_cord, y_cord = bayes_indexes, bayes
plt.plot(x_cord, y_cord, label = 'Naive Bayes', color = 'tab:green')

#DT
x_cord, y_cord = bayes_indexes, dt
plt.plot(x_cord, y_cord, label = 'Decision Tree', color = 'tab:red')

title = "Learning curves"

plt.yticks(np.arange(0.7, 1, 0.05))
plt.ylabel('F1 Score')
plt.xlabel('Number of training data')
plt.grid(True)
plt.title(title)
plt.legend(loc='lower right')
plt.show()
