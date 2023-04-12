import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
IN_DIR = os.path.join(PROJECT_DIR, 'in')
SIMULATED_OUT_DIR = os.path.join(OUT_DIR, 'simulatedood')
REAL_OUT_DIR = os.path.join(OUT_DIR, 'realood')
MODELS_DIR = os.path.join(IN_DIR, 'models')
PLOT_DIR = os.path.join(OUT_DIR, 'figures')

# data paths
DATA_DIR = os.path.join(IN_DIR, 'data', 'ood')
DIST_DATA_DIR = os.path.join(IN_DIR, 'data', 'annotateddist', 'chunks-clip15-30')
DIST_TARGET_NAME = 'targets-ilm-boundary-gauss'

# parameters
RESIZE = (64, 224)
SEQL = 10

# checkpoints and models
EST_CHECKPOINT = os.path.join(MODELS_DIR, '211116-100711_retina-detection/checkpoint_best-AE.pth')
MAHA_FILE = os.path.join(MODELS_DIR, '220223-165711_mahaad/maha.npz')
MAHA_RAW_FILE = os.path.join(MODELS_DIR, '220223-170029_maharaw/maha.npz')
SUP_CHECKPOINT = os.path.join(MODELS_DIR, '220228-231607_supervised/checkpoint_best-loss.pth')
GLOW_CHECKPOINT = os.path.join(MODELS_DIR, '221110-220534_glow/glowINretina2.pt')

# imaging constant
SPACING_MM = 3.7 * 1e-3  # mm
