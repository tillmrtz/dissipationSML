import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import warnings

# Try to use TEOS-10 (gsw) for density; fall back to a crude approx if not available
try:
    import gsw
    HAVE_GSW = True
except Exception:
    HAVE_GSW = False
    warnings.warn("gsw not available — using approximate density formula. Install gsw for accurate seawater thermodynamics.")


# -------------------------
# Helper functions
# -------------------------
def NaN_interp(x):
    """Interpolate interior NaNs linearly, preserve leading/trailing NaNs."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    isn = np.isnan(x)
    if np.all(isn):
        return x.copy()
    good = ~isn
    xi = np.arange(n)
    f = interp1d(xi[good], x[good], bounds_error=False, fill_value=np.nan)
    out = f(xi)
    return out


def denan(x):
    """Remove leading/trailing NaNs and return the 'denaned' vector and indices of non-nans.
       MATLAB denan often returns the trimmed vector and start/end indices; here we return
       vector with all interior NaNs preserved but drop leading/trailing NaNs to produce
       a contiguous chunk suitable for filtering."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    nonan_idx = np.where(~np.isnan(x))[0]
    if nonan_idx.size == 0:
        return np.array([]), np.array([], dtype=int)
    start = nonan_idx[0]
    end = nonan_idx[-1] + 1
    return x[start:end], np.arange(start, end)


def rms(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return np.sqrt(np.nanmean(x ** 2))


def ctr1stdiffderiv(y, t):
    """Centered derivative dy/dt with simple 3-point low-pass smoothing.
       y and t are 1D arrays of same length. We compute gradient and then apply
       a 3-point moving average (to mimic 'centred 1st derivative LPF length 3')."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.size < 2:
        return np.zeros_like(y)
    dy = np.gradient(y, t, edge_order=2)
    # 3-point moving average
    kernel = np.ones(3) / 3.0
    dy_smooth = np.convolve(dy, kernel, mode='same')
    return dy_smooth


def mysmooth(x, window):
    """Simple moving average with integer window (window must be >=1). Return same length."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        out[i] = np.nanmean(seg)
    return out


def sw_dens0(sal, temp):
    """Return density (approx). Prefer gsw.rho if available; else use linear approximation
       (NOT as accurate). Parameters: sal (psu), temp (deg C, potential temperature assumed)."""
    sal = np.asarray(sal)
    temp = np.asarray(temp)
    if HAVE_GSW:
        # TEOS-10 expects Absolute Salinity and Conservative Temperature ideally,
        # but we try practical salinity and in-situ/ptemp conversions approximate:
        try:
            # convert Practical Salinity (SP) to Absolute Salinity (SA) approx via gsw.SA_from_SP
            SA = gsw.SA_from_SP(sal, 0, 0, 0)  # lon/lat/depth=0 fallback
            CT = temp  # if temp is potential temperature, this is an approximation
            rho = gsw.rho(SA, CT, 0.0)  # pressure = 0 dbar approx
            return rho
        except Exception:
            pass
    # fallback approximate formula -- crude, use only if gsw missing
    # baseline 1027.65 at SP=35, T=10 (arbitrary)
    rho0 = 1027.65
    rho = rho0 + 0.2 * (sal - 35.0) - 0.2 * (temp - 10.0)
    return rho


# -------------------------
# Main function
# -------------------------
def apply_LEM_final_down(dives, base, uselayers=True, ell=10.0, Kz=5.0):
    """
    Python translation of apply_LEM_final_down.

    Parameters
    ----------
    dives : iterable of ints
        dive numbers (each typically corresponds to a .mat filename base + %04d.mat)
    base : str
        file base including relative path but excluding the 4-digit dive number, e.g. 'sg_10km2B2/p005'
        full filename used: f"{base}{dive:04d}.mat"
    uselayers : bool
        whether to attempt to categorize layers (requires a Python implementation of catagorize_FBClayers)
    ell : float
        length scale used to compute rms of Thorpe displacements (same units as z)
    Kz : float
        filter length used (same semantics as the MATLAB code)
    Returns
    -------
    RESULT : list of dicts
        each entry contains results for that dive (fields similar to MATLAB struct)
    """

    VMPnoise = 1e-10
    RESULT = []

    for dive in dives:
        print(f"processing dive {dive}")
        fname = f"{base}{dive:04d}.mat"
        try:
            mat = scipy.io.loadmat(fname, struct_as_record=False, squeeze_me=True)
        except FileNotFoundError:
            warnings.warn(f"File not found: {fname} — skipping dive {dive}")
            continue

        # --- Unpack variables expected to be present in the .mat file ---
        # The MATLAB code expects these names: ctd_depth_m, salin, theta (PT_temp),
        # time, w_model, dive_i_corrected, climb_i_corrected, midpoint_day, midpoint_lat, midpoint_lon, etc.
        # If your .mat uses different names, change these mappings here.
        try:
            ctd_depth_m = mat['ctd_depth_m']        # full profile depth vector (array per dive)
            salin = mat['salin']                    # salinity vector (aligned with dive indices)
            theta = mat['theta']                    # potential temperature / PT_temp
            time = mat['time']                      # time array (same length)
            w_model = mat.get('w_model', np.zeros_like(time))  # model vertical velocity if available
            dive_i_corrected = mat['dive_i_corrected']  # indices for dive portion
            climb_i_corrected = mat.get('climb_i_corrected', None)
            midpoint_day = mat.get('midpoint_day', np.nan)
            midpoint_lat = mat.get('midpoint_lat', np.nan)
            midpoint_lon = mat.get('midpoint_lon', np.nan)
            ctd_sg_depth_m = mat.get('ctd_sg_depth_m', ctd_depth_m)
            glideangle_model = mat.get('glideangle_model', None)
            climb_i = mat.get('climb_i', climb_i_corrected)
        except KeyError as ke:
            warnings.warn(f"Missing expected variable {ke} in {fname}. Skipping dive {dive}.")
            continue

        # choose dive and climb indices
        iwd = np.atleast_1d(dive_i_corrected)  # dive indices in file
        iwp = np.atleast_1d(climb_i_corrected) if climb_i_corrected is not None else np.array([])

        # build arrays for dive portion
        z = np.asarray(ctd_depth_m)[iwd]
        PT_temp = np.asarray(theta)
        dens = sw_dens0(np.asarray(salin)[iwd], PT_temp[iwd])
        rho = np.sort(dens)  # nansort equivalent (we assume not many NaNs)
        # timestamp etc.
        idatenum = mat.get('dive', dive)  # fallback
        midtime = midpoint_day
        midlat = midpoint_lat
        midlon = midpoint_lon

        # ensure some data
        if len(iwd) <= 3:
            warnings.warn(f"Dive {dive} has <=3 data points, skipping.")
            continue

        # layer categorization -- user must provide catagorize_FBClayers in Python if uselayers True
        if uselayers:
            try:
                # function must be provided by user; here we assume it exists in the global scope
                layerI = catagorize_FBClayers(PT_temp, ctd_sg_depth_m, glideangle_model, iwd, climb_i)
            except NameError:
                warnings.warn("catagorize_FBClayers not implemented in Python; using zero layers.")
                layerI = np.zeros(len(PT_temp), dtype=int)
        else:
            layerI = np.zeros(len(PT_temp), dtype=int)

        # get w from pressure (ctr1stdiffderiv on negative depth to get vertical velocity)
        w_meas = ctr1stdiffderiv(-np.asarray(ctd_sg_depth_m)[iwd], np.asarray(time)[iwd])
        # subtract model (MATLAB divides w_model by 100 — keep same behavior)
        w = w_meas - np.asarray(w_model)[iwd] / 100.0

        # NaN out points where model is zero
        ind_bad_model = np.where(np.asarray(w_model)[iwd] == 0)[0]
        if ind_bad_model.size > 0:
            w[ind_bad_model] = np.nan

        # remove top ~40 m
        isurface = np.where(z < 40)[0]
        if isurface.size > 0:
            w[isurface] = np.nan

        # Thorpe displacements: need to sort potential temperature and get displacements.
        # MATLAB uses a sort of the non-NaN PT_temp for the dive
        iNaN = ~np.isnan(PT_temp[iwd])
        Tnonan = PT_temp[iwd][iNaN]
        Znonan = z[iNaN]

        if Tnonan.size == 0:
            warnings.warn(f"No valid PT_temp for dive {dive}, skipping.")
            continue

        # sort descending (MATLAB rhosort is sort(Tnonan,'descend'))
        isort = np.argsort(-Tnonan)
        rhosort = Tnonan[isort]
        zsort = Znonan[isort]
        Ltshort = np.abs(Znonan - zsort)

        # remove tiny displacements due to sensor noise
        terror = 0.003
        diff_is_too_small = np.where(np.abs(Tnonan - rhosort) <= terror)[0]
        Ltshort[diff_is_too_small] = 0.0

        # sow Lt into full-length vector
        Lt = np.full_like(z, np.nan, dtype=float)
        Lt_idx = np.where(iNaN)[0]
        Lt[Lt_idx] = Ltshort

        # prepare arrays
        lag_Pass1 = np.full_like(z, np.nan, dtype=float)
        rmsw = np.full_like(z, np.nan, dtype=float)
        gapli = np.full_like(z, np.nan, dtype=float)

        # build absolute-through-water vertical coordinate absZ
        dz = np.abs(np.diff(z))
        if np.any(dz == 0):
            dz[dz == 0] = 0.01
        absZ = np.concatenate(([z[0]], np.cumsum(dz) + z[0]))  # cumsum starting from z[0]

        sampint = np.nanmean(dz)

        # regular depth grid (through-water coordinate)
        zi = np.arange(np.ceil(np.nanmin(absZ)), np.floor(np.nanmax(absZ)) + sampint, sampint)

        # interpolate onto zi
        # interp1 with 'linear' and default extrapolation -> here we supply fill_value=np.nan
        f_w = interp1d(absZ, w, kind='linear', bounds_error=False, fill_value=np.nan)
        wi_wnan = f_w(zi)

        f_rho = interp1d(absZ, rho if np.ndim(rho) == 1 else np.asarray(rho), kind='linear', bounds_error=False, fill_value=np.nan)
        rhosmoothnan = f_rho(zi)

        # remove interior NaNs, preserve bookend NaNs (NaN_interp does that)
        wi = NaN_interp(wi_wnan)
        rhosmoothi = NaN_interp(rhosmoothnan)

        wi_nonan, inan_idx = denan(wi)
        if wi_nonan.size == 0:
            warnings.warn(f"No contiguous non-NaN segment for filtered w in dive {dive}, skipping")
            continue

        # set up high-pass Butterworth filter
        KzSTDY = Kz
        FNorm = KzSTDY / (0.5 * (1.0 / sampint))
        # normalize to Nyquist: FNorm must be <1; if >=1, reduce it slightly
        if FNorm >= 1.0:
            FNorm = 0.9999
        b, a = butter(4, FNorm, btype='high')

        # ensure record length > 3 * filter length
        if wi_nonan.size <= 3 * len(b):
            warnings.warn(f"Dive {dive}: record too short for filtering. skipping.")
            continue

        # filter with zero-phase filtfilt
        try:
            highpass_wi = filtfilt(b, a, wi_nonan, padlen=3*(max(len(a), len(b))-1))
        except Exception:
            # fallback to direct filtfilt without padlen if it fails
            highpass_wi = filtfilt(b, a, wi_nonan)

        # lowpass on density for BVFQ using mysmooth window=3 (per MATLAB)
        lowpass_rho = mysmooth(rhosmoothi, 3)

        # interpolate back to original sample rate z
        # note: zi[inan_idx] maps to the positions in zi that correspond to wi_nonan
        zi_segment = zi[inan_idx]
        f_hp = interp1d(zi_segment, highpass_wi, kind='linear', bounds_error=False, fill_value=np.nan)
        hp_w = f_hp(z)  # map to original z positions
        # ensure we do not overwrite points originally NaN in w
        hp_w[np.isnan(w)] = np.nan

        f_rhosmooth = interp1d(zi, lowpass_rho, kind='linear', bounds_error=False, fill_value=np.nan)
        rhosmooth = f_rhosmooth(z)

        # noise level handling
        wnlvl = 0.002
        hp_w[np.abs(hp_w) < wnlvl] = 0.0

        rmsw_fixlen = np.full_like(z, np.nan, dtype=float)

        # loop compute rms(Lt) and rms(hp_w) for fixed length ell
        for i in range(len(z)):
            inds = np.where(np.abs(z[i] - z) <= ell / 2.0)[0]
            if inds.size > 0:
                rmsLt = rms(Lt[inds])
                rmsw_fixlen[i] = rms(hp_w[inds])
                if rmsLt > 0:
                    lag_Pass1[i] = rmsLt
                else:
                    lag_Pass1[i] = np.nan
            else:
                rmsw[i] = np.nan
                lag_Pass1[i] = ell
                rmsw_fixlen[i] = np.nan

        # velocity scale is absolute of high-pass w
        rmsw = np.abs(hp_w)
        rmsw_fixlen[rmsw_fixlen == 0] = np.nan

        L = lag_Pass1
        # prevent division by zero in L
        e = (rmsw ** 3) / L
        e[np.isinf(e)] = np.nan

        # compute BVFQ using 4-point centered difference like in MATLAB loop (i+2 minus i-2)
        BVFQ = np.full_like(rhosmooth, np.nan, dtype=float)
        for i in range(2, len(rhosmooth) - 2):
            dz2 = z[i + 2] - z[i - 2]
            if dz2 != 0 and not (np.isnan(rhosmooth[i + 2]) or np.isnan(rhosmooth[i - 2])):
                BVFQ[i] = np.sqrt((9.8 / 1027.0) * ((rhosmooth[i + 2] - rhosmooth[i - 2]) / dz2))

        # PeriodRatio and IWinfluence
        with np.errstate(divide='ignore', invalid='ignore'):
            PeriodRatio = (BVFQ * L) / ((2.0 * np.pi) * (rmsw / 100.0))
        IWinfluence = np.where(PeriodRatio > 3)[0]

        # assemble result dict for this dive
        res = {}
        res['dive'] = dive
        res['time'] = midtime
        res['lat'] = midlat
        res['lon'] = midlon
        res['all_time'] = np.full_like(z, midtime, dtype=float)
        res['all_lat'] = np.full_like(z, midlat, dtype=float)
        res['all_lon'] = np.full_like(z, midlon, dtype=float)
        res['rmsw'] = rmsw
        res['rmsw_fixlen'] = rmsw_fixlen
        res['z'] = z
        res['w'] = w
        res['BVFQ'] = BVFQ
        res['T'] = PT_temp[iwd]
        res['rho'] = dens
        res['L'] = L
        res['e'] = e
        res['PeriodRatio'] = PeriodRatio
        res['IWinfluence'] = IWinfluence
        res['hab'] = -(z - np.nanmax(z))
        res['layerI'] = layerI[iwd] if len(layerI) == len(PT_temp) else layerI
        res['gapL'] = gapli
        res['Lt'] = Lt

        # computed z3degLevel based on unique T mapping like MATLAB unique+interp trick
        try:
            uniT, m = np.unique(res['T'], return_index=True)
            # remove NaNs
            ok = ~np.isnan(uniT)
            if np.any(ok):
                interp_fun = interp1d(uniT[ok], res['z'][m[ok]], bounds_error=False, fill_value=np.nan)
                res['z3degLevel'] = float(interp_fun(3.0))
            else:
                res['z3degLevel'] = np.nan
        except Exception:
            res['z3degLevel'] = np.nan

        res['z3'] = res['z3degLevel'] - res['z']
        res['eclean'] = res['e']

        RESULT.append(res)

    return RESULT
