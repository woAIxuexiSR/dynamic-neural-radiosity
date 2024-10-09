import mitsuba as mi
import numpy as np
import argparse

mi.set_variant("cuda_rgb")

EPSILON = 1e-2

# np.ndarray (H, W, 3) -> np.ndarray (H, W)
def compute_MSE(img, ref):
    e = np.square(img - ref)
    return np.mean(e, axis=2)

def compute_relMSE(img, ref):
    e = np.square((img - ref) / (ref + EPSILON))
    return np.mean(e, axis=2)

def compute_MAPE(img, ref):
    e = np.abs(img - ref) / (ref + EPSILON)
    return np.mean(e, axis=2)

def compute_MAE(img, ref):
    e = np.abs(img - ref)
    return np.mean(e, axis=2)

def compute_SMAPE(img, ref):
    e = 2 * np.abs(img - ref) / (img + ref + EPSILON)
    return np.mean(e, axis=2)

def compute_img(img, ref, type):
    if type == "MSE":
        return compute_MSE(img, ref)
    elif type == "relMSE":
        return compute_relMSE(img, ref)
    elif type == "MAPE":
        return compute_MAPE(img, ref)
    elif type == "MAE":
        return compute_MAE(img, ref)
    elif type == "SMAPE":
        return compute_SMAPE(img, ref)
    else:
        raise NotImplementedError
    
def compute_metric(img, ref, type, discard=0.001):
    num = int(img.shape[0] * img.shape[1] * (1 - discard))
    e = compute_img(img, ref, type)
    e = np.sort(e.reshape(-1))[:num]
    return np.mean(e)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare Images")
    parser.add_argument("img", type=str, default="hello.exr")
    parser.add_argument("ref", type=str, default="ref.exr")
    parser.add_argument("metric", type=str, default="MAPE")
    
    args = parser.parse_args()
    
    img = np.array(mi.Bitmap(args.img))
    ref = np.array(mi.Bitmap(args.ref))
    
    value = compute_metric(img, ref, args.metric)
    print("{}: {:.6f}".format(args.metric, value))