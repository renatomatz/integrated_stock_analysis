from dcf import *

with open("key.txt") as f:
    quandl.ApiConfig.api_key = f.readline().strip("\n")

# Stock:
xom = Stock("XOM", rfa="US10Y", mi="SP500", cr="AAA", do_stats=False)

xom.set_rfa(code="US10Y")
xom.set_mi(code="SP500")
xom.set_cr(code="BAA")

print(xom.get_rfa())
print(xom.get_rfa_presets())
print(xom.get_mi())
print(xom.get_mi_presets())
print(xom.get_cr())

# xom.update_beta()
# xom.update_ke()
# xom.update_pgr()
# xom.update_roe()

# xom.do_stats(do_all=True)

xom.beta = 0.606
xom.ke = 0.0612512
xom.pgr = -0.001889855264463426
xom.roe = 0.10772389

# Hamada_UL
hamada_U = Hamada_UL()
# print(hamada_U(hamada_u)(xom))

# Hamada_RL
    
hamada_R = Hamada_RL()
# print(hamada_R(hamada_r)(xom))

# Fernandez_UL
    
fernandez_U = Fernandez_UL()
# print(fernandez_U(fernandez_u)(xom))

# Fernandez_RL
    
fernandez_R = Fernandez_RL()
# print(fernandez_R(fernandez_r)(xom))

# Central_Change_FC
central_fc = Central_Change_FC("fcf", 5)
print(central_fc(np.mean)(xom))

# Mean_Rel
    
# mean_rel = Mean_Rel(col, *args, **kwargs)
# mean_rel(_centre)

# LM_Rel
    
# lm_rel = LM_Rel(y_col, x_col, *args, **kwargs)
# lm_rel(_lm)

# FCFE
fcfe = FCFE(forecast_f=np.mean, forecast_w=central_fc)
print(fcfe.calculate(xom, set_att=True))

# DDM
ddm = DDM(stages=None)
print(ddm.calculate(xom, set_att=True))

# REL # dependent on .get_comps, which is not a standalone function anymore
# rel = REL(mult_f=None, mult_w=None, d_col=None)
# calculate(xom, set_att=False)

# CAPM
capm = CAPM(unlever_f=hamada_u, unlever_w=hamada_U, relever_f=hamada_r, relever_w=hamada_R)
capm = CAPM(unlever_f=fernandez_u, unlever_w=fernandez_U, relever_f=fernandez_r, relever_w=fernandez_R)
print(capm.calculate(xom, set_att=True))

xom.plot_models()

print(p_val(n, r=0, t=1))

print(eq_val_ddm(divs, pgr, ke))
