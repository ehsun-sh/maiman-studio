# Validation

[← README](../README.md) · [Getting started](getting-started.md) · [The studio interface](interface.md) · [Models and results](physics.md) · [Design and roadmap](design.md) · [Validation](validation.md)

---

## Validation

Every physics block ships with a test against a closed-form result, run in CI
([`tests/test_physics.py`](../tests/test_physics.py)):

| Case | Expected | |
| :--- | :--- | :-- |
| Attenuation | `P_out = P_in · 10^(-αL/10)` | ✅ |
| Cascaded spans | Loss is additive in dB | ✅ |
| Source power | Independent of the simulated time window | ✅ |
| Phase noise | Broadens the line, conserves average power | ✅ |
| Multi-carrier | Channels stay separate bands; spacing does not drive `Fs` | ✅ |
| Gaussian pulse, CD only | `T₁/T₀ = √(1 + (z/L_D)²)`, `L_D = T₀²/\|β₂\|` | ✅ |
| Chirped Gaussian | `T₁/T₀ = √((1 + Cβ₂z/T₀²)² + (β₂z/T₀²)²)` — pins the sign of β₂ | ✅ |
| Dispersion compensation | `+D` then `−D` restores the input sample-for-sample | ✅ |
| Receiver-side CD removal | Compensator is the propagator inverted; round trip exact to 1e-9 | ✅ |
| CD compensation is all-pass | Energy conserved; a wrong sign lands exactly on twice the span | ✅ |
| β₂ ∝ λ² | Compensating 1550 nm as 1310 nm leaves the predicted `1 − λ₁²/λ₂²` residual | ✅ |
| **Span recovery** | 5 km to 1000 km return to back-to-back EVM; uncompensated 80 km is at chance | ✅ |
| GVD | Energy conserved (Parseval); β₂ = −Dλ²/2πc per band | ✅ |
| PRBS | Period `2ⁿ−1`; `2ⁿ⁻¹` marks; every n-bit window appears once | ✅ |
| Ideal push-pull MZM | `P_out/P_in = cos²(πV / 2V_π)`; null depth equals the declared ER | ✅ |
| PIN detector | `I = R·P`; shot `σ² = 2qIB`; thermal `σ² = 4kTB/R_L` | ✅ |
| Receiver filter | 3 dB at `B`; noise bandwidth `B·√(π/4ln2)`; zero group delay | ✅ |
| **BER** | `½·erfc(Q/√2)` matched against **directly counted errors**, 10⁻⁴–10⁻¹ | ✅ |
| **SER, every order** | Counted errors against `ser_qam` within 12 %, at 1–8 bits/symbol and 4–22 dB | ✅ |
| Rectangular SER ≡ square SER | The generalised expression reproduces the textbook square formula to 2e-16 at every even order, 0–39 dB | ✅ |
| Odd orders are Gray coded | Every nearest-neighbour pair differs in exactly one bit — which a cross constellation provably cannot manage | ✅ |
| Rotational symmetry is measured | 4 for a square, 2 for a rectangle, read off the point set; narrowing the phase search back to π/2 breaks 32-QAM at a 0.6π offset | ✅ |
| Link consistency | `L` km of span ≡ launching `α·L` dB lower, end to end | ✅ |
| Lossless SSFM | Energy conserved with nonlinearity; γ=0 reproduces the exact linear solution | ✅ |
| Self-phase modulation | `\|A(T)\|` exactly unchanged; spectrum broadens | ✅ |
| **Fundamental soliton (N=1)** | Envelope invariant over 4 soliton periods — **the only test that pins the sign of γ against β₂** | ✅ |
| Higher-order soliton (N=2) | Compresses at half a period, recovers at a full one | ✅ |
| EDFA | `P_ASE = 2·n_sp·hν·(G−1)·B_o`; `n_sp = NF·G/2(G−1)` | ✅ |
| OSNR | `58 + P_launch − NF − 10·log10(spans)`, over 16 spans | ✅ |
| **Signal-ASE beat** | Q on an amplified link tracks `2√(B_ref/B_e)·OSNR/(1+√(1+4·OSNR))` to 15% | ✅ |
| ASE beat, coherent | Electrical SNR converges on `2·OSNR·B_ref/R_s` as ASE dominates — 0.23 dB | ✅ |
| Beat is polarization-selective | Co-polarized ASE beats; orthogonal ASE does not, on both detectors | ✅ |
| **Two carriers beat** | Tone at `Δf` of `2R√(P_a·P_b)` on a diode to 1e-6; none between orthogonal carriers; a neighbour past half the sample rate still adds as a power | ✅ |
| **Delayed self-heterodyne** | A laser against its own delayed copy keeps `exp(−π·Δν·τ)` of the tone: 0.505 for 0.500, 1 at zero delay | ✅ |
| A coherent receiver hears what is in its band | A second carrier 6.25 GHz from the LO lands at `R√(P·P_lo)`, where only the nearest was detected before | ✅ |
| Filter noise bandwidth | `B_n = B·Γ(1+1/2n)/ln2^(1/2n)`, against numerical integration; order 1 is the Gaussian | ✅ |
| Filtered ASE power | Exactly density × `B_n`; a demux passes its own equivalent noise bandwidth | ✅ |
| Wavelength selectivity | A filter between two channels attenuates both; rejection stops at `extinction` | ✅ |
| OSA normalisation | Trace integrates back to an independent power meter; ASE reads density × RBW | ✅ |
| Per-channel OSNR | Survives a demultiplexer that suppresses three channels of four | ✅ |
| Matched filtering | Costs `10·log10(f_s/R_s)` to omit — the receiver integrates noise it cannot use | ✅ |
| **PMD** | DGD Maxwellian: `⟨τ²⟩/⟨τ⟩² = 3π/8`, mean `∝√L`, spread `0.42·mean` | ✅ |
| APD | `F(M) = kM + (2−1/M)(1−k)`; an **interior optimum gain** exists | ✅ |
| **Cross-phase modulation** | `n` equal channels give `(2n−1)×` one channel's nonlinear phase — exact to 1e-3 | ✅ |
| **A neighbour turns a channel's polarization** | A CW pump precesses a probe's Stokes vector about its own by `(4/3)·γPL` with the coherent term and `(2/3)·γPL` phase-only, to 1e-6; a circular pump turns nothing; and against the one-band vector split-step to 7.5e-4, where the probe used to stay put | ✅ |
| **Mixing with the pumps' own phase** | Four amplified spans at 20 mW against the one-band split-step, within 5 % where the linear mismatch is off by 10× and more | ✅ |
| **Cascaded four-wave mixing, within a span** | `cascaded_fwm` lets a first-order product drive a second before the span ends: absent by default (0 W); on, 2.05e-10 W against the one-band split-step's 2.16e-10 W at 10 mW over 20 km, 5 % low where the next term is third order in γ — and the shortfall falls to under 1 % at a tenth the power, where that next term is smaller still. The first-order product is untouched to the last digit | ✅ |
| **Cascaded four-wave mixing, across spans** | `cascaded_fwm` alone does not close this: cutting 20 km into four 5 km spans draws a fresh phase per span for the product-as-pump crossing into the next one, landing 9.7× the split-step. With `carry_phase` too, a product keeps the phase it was made with and the same four spans land at 0.98× at 10 mW and 0.99× at 1 mW, and one 10 km span at 0.99×, 0.97× and 0.89× at 1, 3 and 10 mW as the third-order terms the series leaves out grow; with the mismatch nil the second order grows as the length squared, as it must. Scalar drive; a modulated channel is still drawn. For constant tones `tone_solver` integrates every tone's coupled equations through the span and lands on the split-step to five digits, at 30 mW over 20 km where the series is 25 % off and over lossy amplified spans too; total power holds to 1e-9 without loss. It refuses a modulated band by name | ✅ |
| **Mixing in any state of polarization** | Against the one-band vector split-step, linear, 45°, circular and elliptical, with and without the coherent term: 1.003–1.004, where taking each axis alone gave 0.28 at 45° and 0.58 circular | ✅ |
| The drive is the Kerr tensor's | Isotropic: a linear state mixes as on the axis, a circular one at `(2/3)² = 4/9`; phase-only: `(5/6)² = 25/36` at 45° — and the split-step itself measures 0.4451 and 0.6950 | ✅ |
| **The pumps' Kerr phase in their own state** | 20 mW, every state and both tensors: 0.997–1.001 of the split-step, where per axis the 45° beam was 10.6 % high; four amplified spans at 45° land where the axial ones do, 1.040 and 1.011 | ✅ |
| **Circular and diagonal light over several spans** | Four amplified spans, both tensors: circular and ±45° within 1.003–1.040, as on the axis. The product's phase used to be taken against whichever axis rounding made stronger, which flipped between spans: 1.7× (circular) and 0.12× (−45°) before | ✅ |
| **A turning ellipse, span by span** | Its ellipse turns 6° a span; the geometric phase that adds, and the tensor's phase moving with it, are kept between spans. Cut each span in eight and four spans land at 1.041, beside 1.040 on the axis — where before, cutting drove it from 0.94 to 0.83 | ✅ |
| Elliptical light in one long span | Pinned rather than tuned: 1.09 at D = +4 with the coherent term, within 3 % otherwise, because each state is held along the span | — |
| **Modulated bands on one grid** | `composite_fwm` split-steps the whole comb at once: the tone solver for constants to 5e-4, the one-band split-step over four spans to 2e-3, the time-domain `-i γ L E_b² E_a*` of modulated tones to 1 % — its outer skirt too, at 25 GHz of modulation, past half the spacing. PMD measured from the first band: it comes out as `apply_pmd` alone whatever else is on the grid. Raman as the `T_R` term: two tones on `raman_tilt` to 1e-3 of the change, modulated channels to 0.5 % | ✅ |
| XPM swing | Peak-to-peak `2·γ·P·L_eff` on a probe beside an on/off pump, with no walk-off | ✅ |
| Walk-off | `D·Δλ` per unit length, derived from β₂ and not declared beside it | ✅ |
| Walk-off conserves the mean | Mean XPM phase fixed at `2·γ·⟨P⟩·L_eff` across a 16× change in slip, while its spread falls 5.7× | ✅ |
| FWM efficiency | `η → 1` phase matched; `→ sinc²(Δβ·L/2)` lossless; even in Δβ | ✅ |
| FWM phase mismatch | `Δβ = −β₂(ω_i−ω_k)(ω_j−ω_k)` — quadratic in spacing, zero at zero dispersion | ✅ |
| FWM product power | Component reproduces `d²γ²P_iP_jP_k·L_eff²·η·e^{−αL}` to 1e-7; cubic in power; `d = 2−δ_ij` gives non-degenerate products exactly 6.02 dB | ✅ |
| **Circuit reduction** | Eliminating internal ports agrees with summing round trips lap by lap to 1e-13 — a different algorithm, sharing no code | ✅ |
| Reduction vs SAX | Same models, same wiring: 4.8e-15 over 4001 frequencies. Not in per-push CI — it costs 37 packages and a licence review — but a dispatch-only workflow runs `examples/python/sax_crossvalidation.py` before a release | — |
| Non-reciprocal and reflecting devices | An isolator stays one-way and a mirror returns `r·e^{−2iβL}`; the reduction assumes neither | ✅ |
| Dangling ports | An unwired port is `a = 0`, not a mirror — a 3 dB coupler with one port open passes exactly half | ✅ |
| **Ring resonator** | Assembled from a coupler and two arcs, matches Yariv's all-pass and add-drop transfer functions to 1e-13 | ✅ |
| Free spectral range | `c / (n_g L)` to 1e-4, measured between resonances; `n_eff` moves them and not their spacing | ✅ |
| Critical coupling | Extinction below 1e-12 at `κ = 1 − a²`; under- and over-coupled partners notch identically | ✅ |
| Coupler unitarity | `SᴴS = I` at every split ratio — which is what the cross path's factor of j is for | ✅ |
| Resonance linewidth | Lorentzian `FSR(1−r)/π√r` within 3 % of a measured width from critical coupling to κ = 0.5 | ✅ |
| Waveguide group delay | `n_g L / c` read off the transfer function's phase slope, to 1e-9 | ✅ |
| **Timing estimate** | Tracks a known delay one for one over ±0.45 symbol, to 2e-3 | ✅ |
| Fractional delay is exact | Forward then back returns the input to 1e-12 — a phase ramp, not an interpolation | ✅ |
| Timing recovery earns its place | 500 µm of waveguide costs 675 symbol errors in 1920; with the stage, none, and it moves by the 7.00 ps the guide actually holds | ✅ |
| A whole symbol is invisible | Delay by one symbol period and the estimate does not move — the limit is `\|A\|²`, not the code | ✅ |
| Every shipped project still opens | All six `.maiman` files load, run, and carry their canvas layout — the first thing a new user opens, and nothing checked them before | ✅ |
| **A pilot is an erasure, not an error** | LLR set to zero where the coder's bits were overwritten: 4.9e-4 out of a 9.9e-3 channel, against 2.8e-2 if they are believed | ✅ |
| **A netlist a layout tool wrote solves** | gdsfactory's own shipped sample, unmodified: -0.0053 dB against the 26.637 µm at 2 dB/cm anyone can work out by hand | ✅ |
| Lengths come from the netlist, not the kit | The built waveguide is 10 µm where the kit's nominal is 1000 — asserted on the device, so a lost override says *why* | ✅ |
| A routed circuit resonates where its loop says | FSR 713.74 GHz against `c/(n_g·L)` for the 100.000 µm the instances sum to | ✅ |
| An unmapped cell or port is refused by name | Skipping either returns a circuit that solves and is not the one drawn | ✅ |
| A window narrower than one FSR finds nothing | Kept as a test because a flat 0.995 reads as a ring that does not resonate rather than a scan that is too narrow | ✅ |
| **A ring resonates at two sets of wavelengths** | Two combs, each matching `c/(n_g·L)` for its own group index to a part in a thousand, and their ratio the ratio of the indices | ✅ |
| The combs coincide when the indices do | The control for the above — an off-by-one in the stacking would separate them where there is nothing to separate them | ✅ |
| Stacking reduces exactly to one polarization | Same indices twice reproduces the single-polarization matrix element for element, and the block's output to `rel=1e-12` | ✅ |
| The modes do not leak into each other | Exactly zero, checked at the resonance where circulation would amplify any leak into something visible | ✅ |
| A cross term lands where the solver finds it | Nothing produces one; the matrix can still hold it, so block-diagonal stays a model's choice | ✅ |
| **A junction's heat settles where its closed form says** | The temperature integrated to rest lands on `R_th P_diss`, and its step response gives the time constant back to 5 % | ✅ |
| **Gain dynamics settle onto the static solve** | The reservoir ODE integrated to rest lands on `EDFA.effective_gain` to 1e-9, at seven input powers — two code paths sharing no arithmetic | ✅ |
| The effective time constant is `τ/(1+P_out/P_sat)` | Measured from a step response against the closed form, to a part in a thousand, and monotone in drive | ✅ |
| **A delayed pump loop is stable below a quarter turn** | `a·delay < π/2`: rings down at 0.8 of it, grows at 1.2, and at the edge rings at 1.008 × four delays | ✅ |
| Detector noise wanders the gain by `k·N²/(4τ_c)` | Measured 0.992 of it; the erbium's own lag drops out of a second-order system's variance, so it runs at its real lifetime | ✅ |
| A pump dither is high-passed by the loop | `k·ω/\|a − ω²τ_e + jω\|` to 0.9998 — the quasi-static answer, without the erbium's lag, is 20 % off | ✅ |
| **A PI pump loop follows the linearised step response** | `(kp·s + 1/τ_c)/(τ·s² + (a + kp)·s + 1/τ_c)`: rings at damping 0.07 with no proportional term, near critical by kp = 16, and a first-order loop filter makes it third order — each integrated against its closed form to 2–3 % of the step. Both terms zero, the loop is the old integrator bit for bit | ✅ |
| **A PID pump loop, and anti-windup** | The derivative on the filtered error adds `kd` to the reservoir's inertia: `(kd·s² + kp·s + 1/τ_c)/(τ·τ_f·s³ + (τ + a·τ_f + kd)·s² + (a + kp)·s + 1/τ_c)`, integrated to 3 % of the step. Back-calculation holds an integrator pinned at its limit at `limit − kp·e + T_t·e/τ_c`, to 1e-3 dB, where without it it winds to the limit, and takes a saturating step's overshoot from 0.050 to 0.035 dB | ✅ |
| **An amplifier in the dark holds Lambert's gain** | `W(βG₀)/β` by Halley's iteration against the component's Newton, to 1e-12; its own ASE load is the ASE it emits, both ends; the transient comes to rest there | ✅ |
| **The default fibre is a G.652 fibre once its glass disperses** | Zero dispersion at 1308.3 nm and 16.7 ps/(nm·km) at 1550 from Malitson and Fleming alone; silica's own zero at 1272.7 nm | ✅ |
| A transient is far longer than a window | 25,000 windows at the most saturated point — the measurement the decision to keep it out of the component rests on | ✅ |
| A coarse output grid still gets the right curve | Identical to 1e-6 dB across a 200× range of grid spacing; the integrator takes its own steps and reports how many | ✅ |
| Relaxation is monotone | A first-order system cannot ring, so an overshoot is an integrator bug rather than physics | ✅ |
| **The analyser finds a carrier it was not aimed at** | 1310, 1480, 1550, 1560 and 1625 nm, each located to 0.05 nm with all of its power in the trace — where a fixed window saw nothing outside 1550 | ✅ |
| Full span covers every band, not the loudest | Two carriers 40 nm apart and 20 dB different are both inside the window | ✅ |
| A fixed window still means what it did | `auto_span` off and the 1560 nm carrier is outside the declared span again, exactly as before | ✅ |
| **Acquisition reaches where the fine stage folds** | Band located to ±200 MHz from 0 to ±200 GHz, both signs — 50× past the M-th power's unambiguous range | ✅ |
| The two failure modes are opposites | The fine stage wrong by a whole `symbol_rate/M` at unchanged confidence; the coarse stage's Nyquist wrap correct to derotate by | ✅ |
| No bandwidth is told to it | Roll-off 0 through 1 located identically — a window formulation given 1.6·R_s for a 1.2·R_s band lands 3.9 GHz out | ✅ |
| A flat noise floor does not pull it | Centre unmoved under noise at the signal's own power; only the concentration falls, which is what that number is for | ✅ |
| Acquisition returns a link the fine stage cannot | 4.1, −4.1 and 20 GHz: ~1787 symbol errors without it, zero with it, at the undetuned link's own EVM | ✅ |
| **Pilots resolve every quarter turn** | All four rotations recovered identically and exactly — resolving three of four would make the link work three times in a row and then not | ✅ |
| Pilots are legal symbols, and vary | Drawn from the alphabet's outermost ring, so a pilot is not itself an error, and never constant, so it is not a spectral line | ✅ |
| The estimate reads only the pilots | Every other reference symbol corrupted, and the answer unchanged to 1e-12 — otherwise it is a data-aided estimator in disguise | ✅ |
| Soft information survives the flagship | Real distance to the nearest point and real spread in the LLRs, where the differential decoder gave zero and 1e29 | ✅ |
| **Soft FEC clears what hard FEC cannot** | −25 dBm, 7.4e-3 on the line: the staircase delivers an exact payload where RS(255,239) returns its input | ✅ |
| A staircase stripe row is a codeword | Structural, block by block — the braiding asserted rather than inferred from a curve | ✅ |
| Chase beats its component code | Four errors recovered on a `t=2` code, because they were the least reliable bits | ✅ |
| No competitor means *more* sure, not less | The reliability floor, and the bug it fixes: a negative extrinsic destroys bits the decoder never touched | ✅ |
| A clean block survives the decoder | With varying reliabilities, which is where the broken version failed and a uniform input did not | ✅ |
| The final block is provisional | One error in it is not corrected where the same error anywhere else is — the arrangement, not the code | ✅ |
| Soft cannot be wired into hard | The sixth port type refuses it, because that mistake runs rather than crashes | ✅ |
| **RS(255,239) corrects exactly eight** | Any eight symbol errors anywhere, exactly recovered; nine reported as a failure rather than silently mangled | ✅ |
| The generator's roots are its definition | All sixteen consecutive roots annihilate a derived `g(x)` — the claim, not a transcribed coefficient table | ✅ |
| A burst inside a byte is one error | 64 bad bits in eight bytes correct; 9 bad bits in nine do not — the whole argument for a symbol code | ✅ |
| Closed form matches a counted decode | Bounded-distance expression against Monte Carlo at 3e-3 and 2e-3, within 25 % | ✅ |
| A coded link runs error-free on a broken line | −19 dBm: 7.7e-4 on the line, zero errors in the payload, through real optics and a blind slicer | ✅ |
| Below threshold it degrades *badly* | Post-FEC rate higher than pre-FEC, because miscorrection is what a bounded-distance decoder does past its limit | ✅ |
| **Gain compresses 3 dB at the declared point** | The datasheet definition, checked as a definition, at 10, 20 and 30 dB of gain — the 10 dB row is what catches a dropped `(G₀−2)/G₀` | ✅ |
| The solved gain solves the equation | Newton's answer put back into the implicit Saleh relation, residual under 1e-12 of the small-signal gain | ✅ |
| Compression is smooth, not a ceiling | Gain falls at every step from −40 dBm up and output never stops rising — the two things a clamp gets qualitatively wrong | ✅ |
| Channels share one inversion | 1, 2, 4, 8 channels through a real combiner: 13.61, 12.74, 11.56, 10.12 dBm each, and 8× the input for 5.5 dB more output | ✅ |
| A retired clamp still opens | A project carrying `max_output_power` loads with it dropped rather than refusing | ✅ |
| **RIN-limited SNR** | `1/(RIN·B_n)` to 0.014 dB at −155, −145 and −135 dB/Hz, against the Gaussian filter's closed-form noise bandwidth | ✅ |
| The RIN floor ignores power | Ten times the launch power returns a bit-identical SNR — the one property that makes intensity noise worth modelling | ✅ |
| Intensity noise conserves average power | Moves the variance to `RIN·fs/2` and leaves the mean at the declared dBm, the counterpart of the linewidth invariant | ✅ |
| The two laser noises are independent | Sweeping the linewidth leaves the intensity samples bit-identical, so neither is measuring the other | ✅ |
| **Carrier offset estimate** | Lands on a known offset to 0.5 MHz at both signs across the whole ±3.9 GHz range, on a 32 GBd link | ✅ |
| Frequency recovery earns its place | A 10 MHz LO detuning — 0.03 % of the symbol rate — costs 1677 symbol errors in 1920; with the stage, none | ✅ |
| The stripping power is the geometry's | 8-QAM is refused a quarter turn and stripped at a half, so its range is ±8 GHz and not ±4 | ✅ |
| Past the range it aliases, confidently | Beyond `Rs/2M` the estimate is wrong by exactly `Rs/M` with an unchanged confidence — asserted, not left to be discovered | ✅ |
| No line, no confidence | Circular noise returns an argmax like anything else, and a confidence a fifth of a real one | ✅ |
| The reference design needs both | Detuned 200 MHz, the shipped link runs 3334 errors in 3968; frequency recovery alone leaves 13.0 % EVM, both stages return it to the co-tuned 7.34 % | ✅ |
| **Kernels never touch NumPy** | Every kernel run against an array library that refuses NumPy's *allocating* API and returns identical answers | ✅ |
| A second library gets the same field | `check_device()` propagates an N=1 soliton on each back-end and compares — the one answer that is known without a second run | ✅ |
| The GPU job cannot vanish quietly | A test reads `ci.yml` and holds it to the runner label, the CuPy install and the cross-check | ✅ |
| **Parallel sweeps are invisible** | Bit-identical at 1, 2, 3, 4 and one-per-core workers, in sweep order; derived seeds unchanged | ✅ |
| A failing point does not hang | More points than lanes, so a borrowed graph must come back — verified by deleting the `finally` and watching it deadlock | ✅ |
| **MMI phase relations** | `SᴴS = I` at N = 1, 2, 3, 4, 5, 8 — even amplitudes with invented phases pass every other check and fail this one | ✅ |
| **An uncompensated grating reflects high** | Erdogan's eq. 16 with the pedestal's `σ = 2π·δn̄/λ` to 1e-9; the peak at `λ_B(1 + δn̄/n_eff)`, 0.107 nm at 1e-4; and exactly a compensated grating in a fibre of index `n_eff + δn̄`, to 1e-10 | ✅ |
| A long-period grating's pedestal moves its notches | Each mode lifted by its own self-coupling lands the notch where a mode solve of the raised core puts it: 20 and 60 nm of shift, to half a percent | ✅ |
| A tilted grating's pedestal, two ways | The raised core solved outright against the core mode's first-order self-coupling: 409 against 402.5 pm, the difference halving with the pedestal | ✅ |
| An uncompensated apodized grating chirps itself | The pedestal follows the Gaussian, and the short side grows a Fabry-Perot 140 times the compensated sidelobes while the long side barely moves | ✅ |
| **A grating coupler's own reflection** | Its teeth's second order against the transfer-matrix grating to 1e-9; zero at a half duty cycle; and the fibre's gap ringing at `λ²/2nh` to 5 % | ✅ |
| 2×2 MMI ≡ 3 dB coupler | The same matrix to 1e-15, factor of j included; self-imaging at N = 2 *is* the quadrature relation | ✅ |
| MMI split and imbalance | `1/N` per path; a tilted MMI is lossy and its matrix says so rather than claiming unitarity | ✅ |
| **MZI as a switch** | `sin²(φ/2)` / `cos²(φ/2)` against the assembled circuit, and the sum is 1 across a full turn to 1e-12 | ✅ |
| MZI as an interleaver | `FSR = c / (n_g ΔL)` measured between peaks; `n_eff` would be out by 1.7× | ✅ |
| Balanced MZI has no period | Response flat to 1e-9 across 2 THz — "balanced" means it, not "a period too long to notice" | ✅ |
| **PDK fits** | A polynomial in `λ − λ_ref` [nm], evaluated where you ask; a bare number is a constant | ✅ |
| PDK refuses extrapolation | A C-band coupler fit run to 1310 nm returns **−0.46** — a negative power fraction. `valid_wavelengths` turns that into a sentence | ✅ |
| PDK refuses nonsense | Unregistered component, undeclared parameter, missing cross-section, a cross-section on a lumped device — all at load, all naming the entry | ✅ |
| A kit executes nothing | JSON in, registry lookup out; `"component": "os.system"` is refused like any other unknown name | ✅ |
| Every shipped device builds | At the bottom, middle and top of the kit's own window | ✅ |

Component models are derived from published literature and standards (Agrawal, *Nonlinear Fiber
Optics*; ITU-T G.652 / G.694.1; relevant IEEE 802.3 clauses), cited in each component's
docstring — never from inspection of commercial tools.
