# Simulation Animated Flow

This diagram summarizes how `simulation_animated.py` runs the Monte Carlo crash-risk simulation, builds the animated visualization, and saves the final outputs.

```mermaid
flowchart TD
    startNode["Start script"] --> config["Configuration constants: ZIP_CODE, HOUR, DAY_OF_WEEK, N_TRIALS, RANDOM_SEED"]
    config --> observed["Observed crash data: OBSERVED_CRASHES and OBSERVED_HOURS"]
    observed --> posterior["Posterior parameters: alpha_post = 1 + crashes; beta_post = 1 + hours"]
    posterior --> rngSetup["Initialize random generator and allocate results array"]

    rngSetup --> trialLoop{"For each Monte Carlo trial"}
    trialLoop --> sampleLambda["Sample hourly crash rate: lambda from Gamma posterior"]
    sampleLambda --> sampleWeather["Sample weather condition: clear, rain, or snow"]
    sampleWeather --> adjustRate["Apply weather multiplier: lambda_adj = lambda * multiplier"]
    adjustRate --> drawCrashCount["Draw simulated crash count: Poisson(lambda_adj)"]
    drawCrashCount --> storeResult["Store crash count in results"]
    storeResult --> trialLoop

    storeResult --> metrics["Compute summary metrics"]
    metrics --> crashProbability["p_crash: fraction of trials with at least one crash"]
    metrics --> meanAndCi["mean_crashes and 95 percent interval"]
    metrics --> runningSeries["running_p and running_se convergence series"]

    crashProbability --> styleSetup["Set color palette and Matplotlib rcParams"]
    meanAndCi --> styleSetup
    runningSeries --> styleSetup

    styleSetup --> layout["Create figure and GridSpec layout"]
    layout --> header["Add title and subtitle"]
    layout --> histogramPanel["Build histogram panel: bars, legend, trial counter, live probability text"]
    layout --> convergencePanel["Build convergence panel: reference line, uncertainty band, animated line"]
    layout --> statsPanel["Build stats panel: metric labels and values start transparent"]
    layout --> progressPanel["Build progress bar panel"]

    histogramPanel --> frameSchedule["Create animation frame schedule: log-spaced active frames plus final hold frames"]
    convergencePanel --> frameSchedule
    statsPanel --> frameSchedule
    progressPanel --> frameSchedule

    frameSchedule --> updateCallback["update(fi) animation callback"]
    updateCallback --> updateHistogram["Update histogram bar heights"]
    updateCallback --> updateConvergence["Update convergence line through current trial"]
    updateCallback --> updateProgress["Update progress bar and label"]
    updateCallback --> fadeStats["Fade in final stats near the end"]
    updateHistogram --> returnArtists["Return changed artists for blitting"]
    updateConvergence --> returnArtists
    updateProgress --> returnArtists
    fadeStats --> returnArtists

    returnArtists --> funcAnimation["Create FuncAnimation: 150 frames at about 18 fps"]
    funcAnimation --> saveGif["Save simulation_animated.gif"]
    saveGif --> savePng["Render final frame and save simulation_output.png"]
    savePng --> showPlot["Show Matplotlib window"]
```

## Notes

The script runs top-to-bottom: it performs the full Monte Carlo simulation before any animation frames are rendered. The animation then replays progressively larger prefixes of the already-computed `results` array.

`update(fi)` is the central animation callback. It reads the scheduled trial count for frame `fi`, refreshes the histogram, convergence line, progress bar, live probability label, and fades in the final stats during the hold frames.
