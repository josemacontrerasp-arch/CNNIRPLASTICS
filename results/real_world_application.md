# Real-World Application of FTIR Plastic Classification

Honest assessment of whether this project would work in the real world,
how FTIR is actually used in industry, and where the model realistically
fits.

---

## Short answer

The conveyor-belt picture you imagine **mostly already exists in
industry, but it does not use FTIR.** It uses **NIR (near-infrared)
hyperspectral cameras.** Our project, as built, would not run on a
recycling-plant conveyor. But that does not make it useless. It fits a
different real-world job that genuinely matters.

---

## What you are picturing vs. how recycling plants actually do it

### The industry standard: NIR hyperspectral sorting (already deployed)

This is what is running right now in big recycling plants. Companies
like **Tomra (Autosort)**, **Pellenc ST (Mistral)**, **Steinert
UniSort**, **Sesotec**.

```
                  NIR light source
                        |
                        v
       +------------------------------+
       |  item  item  item  item      |  <- conveyor, 1-3 m wide, ~3 m/s
       +------------------------------+
                        ^
                  hyperspectral
                  line camera (1024 px x 256 wavelengths, ~100 lines/sec)
                        |
                        v
                  per-pixel classifier
                        |
                        v
                  air jet ejects the
                  PET / HDPE / PVC / etc.
                  into the right bin
```

- Camera looks at the whole belt **continuously**, not one sample at a
  time.
- Each pixel of each line gets its own small spectrum (~256 wavelength
  bands in the 1000-1700 nm range).
- A classifier (same idea as our CNN, often simpler) labels each pixel.
- Pneumatic jets fire ~10 ms later and knock the labeled items off
  the belt.
- Real-world throughput: **8-15 tonnes of mixed plastic per hour**,
  95-98 % purity on PET / HDPE / PP streams.

This works because:

1. **NIR (1000-2500 nm) is a longer wavelength** than mid-IR. Its
   light penetrates plastic at a useful depth, reflects off the
   surface, and the camera reads that reflection from ~30 cm away.
   No contact, no sample prep.
2. **Hyperspectral line cameras read 1000+ lines per second.** That
   matches conveyor speed.

### Why mid-IR FTIR (our project) does NOT work on a recycling conveyor

FTIR in the lab works by either:

- Pressing the sample against an **ATR crystal** (diamond / germanium)
  and reading the spectrum through evanescent waves. Physical contact
  required. Each measurement takes ~10-60 seconds.
- **Transmission** through a thin sample on a window. Even slower, and
  you need to prepare the sample (KBr pellet, thin film) first.

On a moving conveyor:

- No contact possible. The sample is whizzing by. ATR is dead.
- Mid-IR does not reflect well off plastic surfaces at standoff
  distance. Most of the signal is absorbed or scattered, not bounced
  back at the detector.
- Scan time per spectrum is seconds, not milliseconds. Throughput
  would be ridiculous.
- Dust, humidity, and IR atmospheric bands (CO2, water vapor) wreck
  open-air mid-IR more than they do NIR.

So our model **as it is** would not classify plastics on a conveyor.
It was trained on stationary, lab-quality FTIR spectra.

---

## Where FTIR (and our model) genuinely belong in the real world

FTIR is not industrial-conveyor technology, but it has real, important
jobs that NIR cannot do.

### 1. Black and dark-colored plastics: the famous gap

NIR fails on **black plastics** because they contain carbon black
pigment, which absorbs almost all NIR light. The reflected spectrum is
basically zero, classifier blind. About **30 % of automotive and
electronics plastic waste is black.** This currently gets dumped to
landfill in huge volumes.

Mid-IR FTIR can still identify black plastics because the absorption
peaks come from molecular vibrations, not bulk reflection. Several
startups are building **mid-IR / quantum-cascade-laser sorting
systems** specifically for this gap (e.g. QuantaSpec, NIR-Online's
hybrid units). A CNN like ours, retrained on mid-IR reflectance data,
could plug straight into one of these.

### 2. Microplastics analysis: this is huge

Honestly, this is where work like ours has the most real-world impact
today. Microplastics (< 5 mm fragments in water, soil, sediment,
marine organisms, human tissue) cannot be sorted on a conveyor. They
are too small to handle physically. They get identified one particle
at a time under a **microscope coupled to an FTIR (uFTIR)** or imaging
FTIR (FPA-FTIR). A technician currently does this by hand, manually
comparing each particle's spectrum against a library. Incredibly slow.

A trained CNN replaces the human in that loop. This is an **active
research area** and there are publications doing exactly our project's
pipeline (1D CNN on FTIR spectra) for microplastic classification.
See: Zavala lab at UW-Madison (the GitHub link cited in our training
script), groups at Karlsruhe, Plymouth Marine Lab, NOAA.

### 3. Quality control in plastic manufacturing

Mid-IR FTIR is the standard QC technique for **incoming raw materials**
at plastic processors. Every batch of resin pellets gets sampled and
run through an FTIR to verify what the supplier sent. A trained
classifier could automate that ten-second check.

### 4. Forensics and customs

Identifying plastic in evidence bags, counterfeit goods, food contact
materials. Slow, careful, lab-based. Same pipeline as ours.

### 5. Field-portable mid-IR spectrometers

Devices like the **Agilent 4300 Handheld FTIR** and the **Bruker Alpha
II** exist. Small enough to take to a recycling line or a beach
cleanup. Slower than conveyor speed (seconds per measurement) but
field-deployable. A CNN running on a laptop or phone attached to one
of these is a realistic deployment.

---

## What our project could realistically become

| Use case | What we'd need to add | Realistic? |
|---|---|---|
| **Microplastics in environmental samples** | More FTIR data on weathered / mixed / colored samples; better out-of-distribution detection; calibration on uFTIR microscope data | **Yes. This is where work like ours lives in published research today.** |
| **Black-plastic conveyor sorting** | Retrain on mid-IR *reflectance* data (not transmission); fast detector; partner with a hardware vendor | Yes, but engineering-heavy. Active commercial frontier. |
| **Handheld-FTIR field tool** | Wrap the model in an app; connect to an Agilent / Bruker portable; handle their data format | Yes, totally doable as a student project. |
| **Replacing NIR conveyor sorting** | Different sensor (NIR camera, not FTIR), retrain entirely on NIR data, rewrite for pixel-wise classification | No. Wrong tool. NIR systems are mature; we would be re-inventing them with the wrong sensor. |

---

## Honest pitch for the presentation

If anyone in the audience asks *"would this actually work in the real
world?"*:

> **"Not on a recycling-plant conveyor. That job is already done by
> NIR hyperspectral cameras, which scan a whole belt at line speed and
> do not need contact. FTIR is too slow and needs the sample in
> contact with the instrument. But FTIR is the right tool for two
> things NIR cannot do: black plastics, where NIR is blind because
> carbon pigments absorb the signal, and microplastics analysis, where
> particles are too small to sort physically and have to be identified
> one by one under a microscope. Our model is most realistically a
> step toward automating microplastic identification, replacing a slow
> manual library-matching workflow that researchers and environmental
> labs currently do by hand."**

That answer is honest, shows you understand the landscape, and does
not overpromise. It also reframes the work as useful in a real domain
rather than "a cool exercise that does not ship anywhere."

---

## Closing reality check

Even within the microplastics use case, going from our current model
to a fielded tool would need:

- **Spectra of weathered polymers** (UV-degraded plastic looks
  different from fresh plastic). Our current training data is mostly
  virgin polymer.
- **Mixed-polymer detection.** Real microplastics often have surface
  contamination, pigments, plasticizers. Our model would currently
  force-classify a mixed particle into one of 6 buckets.
- **The "I don't know" output** (also raised on slide 12). Without
  it, a cellulose fiber would be confidently labeled as PET. Not
  acceptable for environmental work.
- **Validation against the ground truth.** Researchers in this field
  demand independent confirmation (pyrolysis-GC-MS, Raman, density
  separation) before trusting a classifier.

These are next steps, not blockers. The project is at a sensible
stage for an undergraduate or Master's research project. It has
demonstrated the core idea works (CNN classifies FTIR spectra at
~99 % on the test set, generalizes across instruments), and the
remaining work is real but achievable.
