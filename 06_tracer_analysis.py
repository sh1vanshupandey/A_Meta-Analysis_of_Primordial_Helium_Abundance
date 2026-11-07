#File : 06_tracer_analysis.py

import json,os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import comb
from scipy.stats import cauchy,kstest,laplace,norm,t as student_t

os.makedirs("outputs/tables",exist_ok=True)
os.makedirs("outputs/processed",exist_ok=True)

print("="*70+"\nLoading Cleaned Dataset for Tracer-Wise Analysis\n"+"="*70)
df=pd.read_csv("outputs/processed/helium_cleaned.csv")
tracer_counts=df["Tracer"].value_counts()
print("\nObservational Tracer Counts in Cleaned Dataset:")
print(tracer_counts)

# Gott binomial median uncertainty
def calculate_gott_median_uncertainty(data,alpha=0.3173):
    """Calculate median and its binomial confidence interval."""
    sorted_d=np.sort(np.asarray(data,float))
    n=len(sorted_d)
    if n==0:return np.nan,np.nan,np.nan
    median_val=np.median(sorted_d)
    if n==1:return median_val,0.0,0.0
    p_binom=np.array([comb(n,i)*(0.5**n) for i in range(n+1)])
    cumsum=np.cumsum(p_binom)
    left_idx=np.argmin(np.abs(cumsum-alpha/2))
    right_idx=np.argmin(np.abs(cumsum-(1-alpha/2)))
    left_idx=max(0,min(left_idx,n-1))
    right_idx=max(0,min(right_idx,n-1))
    lower_bound,upper_bound=sorted_d[left_idx],sorted_d[right_idx]
    return median_val,median_val-lower_bound,upper_bound-median_val

# Calculate central estimators per tracer
tracer_estimates={}
for tracer in tracer_counts.index:
    subset=df[df["Tracer"]==tracer]
    Y_vals=subset["Y"].dropna().to_numpy()
    N_sub=len(Y_vals)
    if N_sub==0:continue

    # Arithmetic mean and standard error
    am_val=Y_vals.mean()
    sigma_am=np.sqrt(np.sum((Y_vals-am_val)**2)/N_sub**2) if N_sub>1 else 0.0

    # Weighted mean and standard error
    weighted_sub=subset[subset["sigma"]>0]
    if len(weighted_sub)>0:
        weights=1.0/(weighted_sub["sigma"]**2)
        wm_val=np.sum(weighted_sub["Y"]*weights)/weights.sum()
        sigma_wm=np.sqrt(1.0/weights.sum())
    else:
        wm_val,sigma_wm=np.nan,np.nan

    # Median and Gott binomial uncertainty
    med_val,err_minus,err_plus=calculate_gott_median_uncertainty(Y_vals)
    tracer_estimates[tracer]={
        "N":N_sub,"N_weighted":len(weighted_sub),
        "mean":am_val,"sigma_mean":sigma_am,
        "wmean":wm_val,"sigma_wmean":sigma_wm,
        "median":med_val,"sigma_minus":err_minus,"sigma_plus":err_plus
    }

# Save tracer central estimates
summary_rows=[]
for tracer,r in tracer_estimates.items():
    summary_rows.append({
        "Tracer":tracer,"N":r["N"],
        "Arithmetic_Mean":r["mean"],"Arithmetic_Mean_Error":r["sigma_mean"],
        "Weighted_Mean":r["wmean"],"Weighted_Mean_Error":r["sigma_wmean"],
        "Median":r["median"],"Median_Error_Lower":r["sigma_minus"],
        "Median_Error_Upper":r["sigma_plus"]
    })
df_estimates=pd.DataFrame(summary_rows)
df_estimates.to_csv("outputs/tables/tracer_central_estimates.csv",index=False)

# Calculate tracer-wise normalized residuals
print("\nCalculating Tracer-Wise Normalized Residuals...")
df["Tracer_Residual_Mean"]=np.nan
df["Tracer_Residual_Weighted_Plus"]=np.nan
df["Tracer_Residual_Weighted_Minus"]=np.nan
df["Tracer_Residual_Weighted"]=np.nan  # Alias for wm-
df["Tracer_Residual_Median"]=np.nan

for tracer,r in tracer_estimates.items():
    idx=df.index[df["Tracer"]==tracer]

    # Arithmetic mean residuals
    denom_am=np.sqrt(df.loc[idx,"sigma"]**2+r["sigma_mean"]**2)
    valid_am=denom_am>0
    df.loc[idx[valid_am],"Tracer_Residual_Mean"]=(
        df.loc[idx[valid_am],"Y"]-r["mean"]
    )/denom_am[valid_am]

    # Weighted mean residuals: wm+ and wm-
    if not np.isnan(r["wmean"]):
        # wm+: uncorrelated residual
        denom_wm_plus=np.sqrt(df.loc[idx,"sigma"]**2+r["sigma_wmean"]**2)
        valid_wm_plus=denom_wm_plus>0
        df.loc[idx[valid_wm_plus],"Tracer_Residual_Weighted_Plus"]=(
            df.loc[idx[valid_wm_plus],"Y"]-r["wmean"]
        )/denom_wm_plus[valid_wm_plus]

        # wm-: correlated residual; do not clip invalid denominators
        denom_wm_minus_sq=df.loc[idx,"sigma"]**2-r["sigma_wmean"]**2
        valid_wm_minus=denom_wm_minus_sq>0
        df.loc[idx[valid_wm_minus],"Tracer_Residual_Weighted_Minus"]=(
            df.loc[idx[valid_wm_minus],"Y"]-r["wmean"]
        )/np.sqrt(denom_wm_minus_sq[valid_wm_minus])

        # Backward-compatible alias for wm-
        df.loc[idx[valid_wm_minus],"Tracer_Residual_Weighted"]=(
            df.loc[idx[valid_wm_minus],"Y"]-r["wmean"]
        )/np.sqrt(denom_wm_minus_sq[valid_wm_minus])

    # Median residuals
    err_med=np.where(
        df.loc[idx,"Y"]>=r["median"],r["sigma_plus"],r["sigma_minus"]
    )
    denom_med=np.sqrt(df.loc[idx,"sigma"]**2+err_med**2)
    valid_med=denom_med>0
    df.loc[idx[valid_med],"Tracer_Residual_Median"]=(
        df.loc[idx[valid_med],"Y"]-r["median"]
    )/denom_med[valid_med]

# Kolmogorov-Smirnov testing functions
def symmetrize_residuals(x):
    clean_x=x[np.isfinite(x)]
    return np.concatenate([-clean_x,clean_x])

def ks_test_dist(r,dist_name,S,df_t=None):
    scaled_r=r/S
    if dist_name=="Gaussian":
        return kstest(scaled_r,norm.cdf)
    elif dist_name=="Laplace":
        return kstest(scaled_r,laplace.cdf)
    elif dist_name=="Cauchy":
        return kstest(scaled_r,cauchy.cdf)
    elif dist_name=="Student's-t":
        if df_t is None:raise ValueError("Degrees of freedom required for Student's-t.")
        return kstest(scaled_r,lambda val:student_t.cdf(val,df=df_t))
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

def optimize_scale(r,dist_name,df_t=None):
    res=minimize_scalar(
        lambda S:-ks_test_dist(r,dist_name,S,df_t).pvalue,
        bounds=(0.05,5.0),method="bounded"
    )
    S_opt=res.x
    result=ks_test_dist(r,dist_name,S_opt,df_t)
    return S_opt,result.statistic,result.pvalue

def run_ks_suite(r):
    out={}
    # Gaussian, Laplace and Cauchy
    for dist in ["Gaussian","Laplace","Cauchy"]:
        S_opt,D_opt,p_opt=optimize_scale(r,dist)
        res_s1=ks_test_dist(r,dist,1.0)
        out[dist]={
            "S1_D":float(res_s1.statistic),
            "S1_pvalue":float(res_s1.pvalue),
            "S":float(S_opt),"D":float(D_opt),"p":float(p_opt)
        }

    # Student's-t: optimize scale and degrees of freedom
    best_t=None
    for nu in range(2,51):
        S_opt,D_opt,p_opt=optimize_scale(r,"Student's-t",nu)
        if best_t is None or p_opt>best_t[2]:
            best_t=(S_opt,D_opt,p_opt,nu)

    res_s1_t=ks_test_dist(r,"Student's-t",1.0,df_t=best_t[3])
    out["Student's-t"]={
        "S1_D":float(res_s1_t.statistic),
        "S1_pvalue":float(res_s1_t.pvalue),
        "S":float(best_t[0]),"D":float(best_t[1]),
        "p":float(best_t[2]),"df":int(best_t[3])
    }
    return out

# Run KS tests for tracer groups with N>10
tracer_ks_results={}
for tracer,r in tracer_estimates.items():
    if r["N"]<=10:continue
    g=df[df["Tracer"]==tracer]
    tracer_ks_results[tracer]={}
    cols=[
        ("mean","Tracer_Residual_Mean"),
        ("wmean_plus","Tracer_Residual_Weighted_Plus"),
        ("wmean_minus","Tracer_Residual_Weighted_Minus"),
        ("median","Tracer_Residual_Median")
    ]
    for est_name,col_name in cols:
        raw_res=g[col_name].dropna().to_numpy()
        if len(raw_res)>=3:
            tracer_ks_results[tracer][est_name]=run_ks_suite(
                symmetrize_residuals(raw_res)
            )

# Save results
with open("outputs/processed/tracer_ks_results.json","w") as f:
    json.dump(tracer_ks_results,f,indent=4)

ks_summary_rows=[
    {
        "Tracer":tracer,"Estimator":est,"Distribution":dist,
        "S1_D":v["S1_D"],"S1_pvalue":v["S1_pvalue"],
        "S* (Scale)":v["S"],"D (K-S Stat)":v["D"],
        "p-value":v["p"],"df (nu)":v.get("df",np.nan)
    }
    for tracer,es in tracer_ks_results.items()
    for est,ds in es.items()
    for dist,v in ds.items()
]
pd.DataFrame(ks_summary_rows).to_csv(
    "outputs/tables/tracer_ks_summary.csv",index=False
)
df.to_csv("outputs/processed/tracer_residuals.csv",index=False)

# Console summaries
print("\n"+"="*70+"\nTRACER-WISE ESTIMATORS SUMMARY\n"+"="*70)
print(df_estimates.to_string(index=False))
print("\n"+"="*70+"\nTRACER-WISE K-S BEST-FIT RESULTS\n"+"="*70)

for tracer,es in tracer_ks_results.items():
    print(f"\nTracer: {tracer}\n"+"-"*50)
    for est,ds in es.items():
        best_dist=max(ds,key=lambda x:ds[x]["p"])
        v=ds[best_dist]
        df_str=f" (df={v['df']})" if best_dist=="Student's-t" else ""
        print(
            f"  {est.upper():<11} Estimator -> Best Fit: {best_dist}{df_str} |"
            f" p-value = {v['p']:.6f} (S*={v['S']:.4f})"
        )

print("\n"+"="*70+"\n06_tracer_analysis.py Completed Successfully\n"+"="*70)
