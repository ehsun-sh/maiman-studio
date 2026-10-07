---
title: "Array back-ends"
description: "NumPy everywhere, CuPy for the propagation kernels, and why there is no FFTW."
---

# Array back-ends

The propagation kernels are the only part of the engine worth a GPU: a loop over FFTs on a long
array. Everything else is scalar arithmetic or a closed form evaluated a few dozen times, and a
device would slow it down.

## Checking your machine

```
maiman devices
```

```
Array back-ends
  numpy        available
  cupy         not installed

Only NumPy here, so there is nothing to cross-check.
A GPU needs `pip install cupy-cuda12x` and a CUDA device.
```

NumPy is enough for everything in these docs. A GPU only matters for long split-step runs. Where
CuPy and a device are present, `maiman devices` propagates an N = 1 soliton on each back-end and
compares them: the one answer known without a second run.

## The arrays decide, not a setting

A kernel handed CuPy arrays runs on CuPy and returns CuPy arrays; handed NumPy arrays it runs on
NumPy. There is no global mode and no flag on the context. That matters because the kernels are
pure functions, and a hidden mode would be the one piece of state that could make the same inputs
give different answers. Dispatch is on the array's own type, so there is no registry to keep in
sync.

## The contract is a fixed set of names

Every kernel is run against a second array library that refuses NumPy's *allocating* API
(functions like `np.zeros` and `np.fft.fftfreq` that build on the host and would pay a transfer
every step inside a loop). The names that library was asked for are recorded, and the set is
asserted as an equality, not a lower bound, so a change that reaches for something only NumPy has
fails in the repository rather than on somebody's GPU, and one that stops needing something fails
too. CuPy provides all of them.

## What is converted, and what is tested

Only the propagation path. The four-wave-mixing closed forms and the 2×2 Jones algebra stay on the
host, and a test names which functions are in and which are out, so the line is a decision rather
than an oversight.

**CuPy itself is not exercised in CI**, because no runner has a device. What is tested is the half
that would actually break a port, the interface, and a CI job runs the cross-check on any runner
labelled `gpu`. A test reads the workflow file and holds it to that label, the CuPy install and the
check, so the job cannot vanish quietly.

## Why no FFTW

Around 90–95 % of a split-step run is inside the FFT, which is library code in any language: the
reason the engine is Python behind a narrow kernel boundary rather than C++. FFTW would be the
obvious library, and it is GPL-2.0-or-later: linking it, or pyFFTW, would make the whole project
GPL. NumPy's pocketfft, which is BSD, is used instead. The kernel boundary leaves room for a user
to opt into another FFT in their own GPL-compatible deployment.
