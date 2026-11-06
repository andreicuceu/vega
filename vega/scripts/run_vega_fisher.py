"""INI adapter for the notebook Fisher engine."""

import argparse

from vega.fisher import compute_fisher
from vega.vega_interface import VegaInterface


def run_fisher(config_path, output=None):
    """Read normal Vega INIs plus [fisher], compute and save a FisherResult."""
    vega = VegaInterface(config_path)
    if "fisher" not in vega.main_config:
        raise ValueError("The main INI must contain a [fisher] section.")
    config = vega.main_config["fisher"]
    allowed = {
        "parameters",
        "steps",
        "fiducial",
        "output",
        "include-priors",
        "check-steps",
        "step-tolerance",
        "rcond",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown [fisher] options: {sorted(unknown)}")
    names = config.get("parameters", " ".join(vega.sample_params["limits"])).split()

    def values(option, required):
        raw = config.get(option, "").split()
        if not raw and not required:
            return None
        if len(raw) != len(names):
            raise ValueError(f"[fisher] {option} must have one value per parameter.")
        return dict(zip(names, map(float, raw)))

    steps = values("steps", True)
    fiducial = values("fiducial", False)
    destination = output or config.get("output")
    if not destination:
        raise ValueError("Specify [fisher] output or --output.")
    result = compute_fisher(
        vega,
        names,
        steps=steps,
        fiducial=fiducial,
        priors=None if config.getboolean("include-priors", True) else {},
        check_steps=config.getboolean("check-steps", True),
        step_tolerance=config.getfloat("step-tolerance", 0.05),
        rcond=config.getfloat("rcond", 1e-10),
    )
    result.save(destination)
    print(f"Fisher rank {result.rank}/{len(names)}; saved {destination}")
    return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run a fixed-covariance Vega Fisher forecast.")
    parser.add_argument("config", help="Main Vega INI with a [fisher] section")
    parser.add_argument("--output", help="Override [fisher] output NPZ path")
    args = parser.parse_args()
    run_fisher(args.config, args.output)


if __name__ == "__main__":
    main()
