# Video 3 - Building the Simulation


## Setup and why Monte Carlo

In the last video, we got a clean feature table - crash counts bucketed by ZIP code, hour, and day of week, with weather merged in from NOAA. Our job in this video is to take that table and produce a single number: the probability of at least one crash happening in a specific hour.

The naive approach would be to divide total crashes by total hours and call it a rate. But that ignores two real sources of uncertainty - we don't know the true crash rate with confidence, and we don't know what the weather will be. Monte Carlo handles both at once, and that's what we're building today.

## Gamma and Poisson

Before we touch the code, two distributions in 90 seconds.

First, the Gamma distribution. We observed 42 crashes across 52 Tuesday 5 PM hours in ZIP 10003. That gives us a best guess at the rate - but not certainty. The Gamma is a curve over all the plausible values of that rate. It's wide when we have little data, narrow when we have a lot. In the simulation, every single trial draws a different λ from this curve. That's how we capture estimation uncertainty - instead of one fixed rate, we use the whole range of possibilities.

Second, the Poisson distribution. Given a crash rate λ, how many crashes actually show up in one hour? Crashes arrive randomly and independently, so their count follows a Poisson. It gives you the probability of exactly 0, 1, 2, 3 crashes. The key bar to notice is k equals zero - that's the no-crash outcome.

The two connect like this: sample a λ from the Gamma, plug it into the Poisson, get a crash count. That's one trial. Now do that 10,000 times.

## Flowchart walkthrough

Here's the full loop as a flowchart before we look at the code.

We start with N equals 10,000. Inside the loop, four steps every trial: sample λ from the Gamma posterior, sample a weather condition from NOAA probabilities, multiply λ by a weather adjustment factor, then draw k from Poisson of λ-adjusted. Store k, repeat. After all trials are done, we aggregate - the crash probability is just the fraction of trials where k was at least 1.

## Code walkthrough

The code follows this exactly. Let me walk through it in blocks.

Block 1 - configuration. We set the scenario: ZIP 10003, Tuesday 5 PM. Two numbers come from Vic's feature table: 42 observed crashes, 52 observed hours in that slot. Those feed directly into the posterior.

Block 2 - the Gamma posterior. We use a conjugate Bayesian update. The prior is weakly informative - Gamma(1, 1). After seeing the data, the posterior becomes Gamma(1 plus 42, 1 plus 52) - that's Gamma(43, 53). The mean of that is 43 over 53, about 0.81 crashes per hour. But instead of locking in that single number, we sample from the full distribution every trial.

Block 3 - the loop. Seven lines. Each iteration: draw λ from the Gamma posterior, pick a weather condition weighted by NOAA probabilities - clear 65%, rain 28%, snow 7% - multiply λ by the weather multiplier, draw from Poisson, store the result. That's it.

Block 4 - aggregate. P(crash) equals the number of trials with k at least 1, divided by 10,000.

## The Output

And here's what it produces.

The left panel is the histogram of crash counts across all 10,000 trials. Green is k equals zero - no crash. Red is k at least one - a crash happened. The P(crash) counter in the middle updates live as trials accumulate.

Notice the right panel - the convergence diagnostic. Early on, with only 10 or 20 trials, the estimate bounces wildly. [Point to the chaotic early section.] That's just noise - too few samples. As we push past a few hundred trials it starts settling, and by 10,000 it's essentially flat at 59.8%. The dashed red line is the final converged estimate.

This is the law of large numbers playing out in real time. It also tells us that 10,000 trials is more than enough - running a million wouldn't meaningfully change the answer.

What it doesn't tell us is whether 59.8% is actually calibrated - whether predictions at that level come true roughly 60% of the time in the real world. That's the validation problem, and that's exactly what is tackled in the next video.