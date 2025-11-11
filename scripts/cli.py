"""Unified CLI for VIX ETP RL training, tuning, and evaluation."""

import argparse
import sys
from pathlib import Path

from scripts.train_eval_pipeline import train_agent, evaluate_agent
from scripts.optuna_penalty_tune import tune_hyperparameters
from scripts.rolling_walkforward import rolling_walkforward
from scripts.generate_report import generate_report, print_summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='VIX ETP RL Agent CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a PPO agent
  python scripts/cli.py train --agent ppo --timesteps 100000
  
  # Tune hyperparameters
  python scripts/cli.py tune --trials 50
  
  # Run walk-forward validation
  python scripts/cli.py walkforward --agent ppo --timesteps 50000
  
  # Generate report
  python scripts/cli.py report
  
  # Evaluate a trained model
  python scripts/cli.py eval --model outputs/vix_etp_ppo_model.zip --agent ppo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                             help='Path to config file')
    train_parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo',
                             help='Agent type')
    train_parser.add_argument('--timesteps', type=int, default=100000,
                             help='Total training timesteps')
    train_parser.add_argument('--output', type=str, default='outputs',
                             help='Output directory')
    train_parser.add_argument('--seed', type=int, default=None,
                             help='Random seed')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained agent')
    eval_parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                            help='Path to config file')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo',
                            help='Agent type')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--output', type=str, default='outputs',
                            help='Output directory')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Tune hyperparameters with Optuna')
    tune_parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                            help='Path to config file')
    tune_parser.add_argument('--trials', type=int, default=None,
                            help='Number of trials (default: from config)')
    tune_parser.add_argument('--timesteps', type=int, default=50000,
                            help='Training timesteps per trial')
    tune_parser.add_argument('--output', type=str, default='outputs',
                            help='Output directory')
    tune_parser.add_argument('--timeout', type=int, default=None,
                            help='Timeout in seconds')
    tune_parser.add_argument('--jobs', type=int, default=None,
                            help='Number of parallel jobs')
    
    # Walkforward command
    wf_parser = subparsers.add_parser('walkforward', help='Run rolling walk-forward validation')
    wf_parser.add_argument('--config', type=str, default='config/vix_etp_config.yaml',
                          help='Path to config file')
    wf_parser.add_argument('--train-window', type=int, default=None,
                          help='Training window in days')
    wf_parser.add_argument('--test-window', type=int, default=None,
                          help='Test window in days')
    wf_parser.add_argument('--step', type=int, default=None,
                          help='Step size in days')
    wf_parser.add_argument('--agent', type=str, choices=['ppo', 'sac'], default='ppo',
                          help='Agent type')
    wf_parser.add_argument('--timesteps', type=int, default=50000,
                          help='Training timesteps per fold')
    wf_parser.add_argument('--output', type=str, default='outputs',
                          help='Output directory')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--results-dir', type=str, default='outputs',
                              help='Directory containing results')
    report_parser.add_argument('--output', type=str, default='vix_etp_report.png',
                              help='Output file for plots')
    report_parser.add_argument('--text-only', action='store_true',
                              help='Print text summary only (no plots)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'train':
            print(f"Training {args.agent.upper()} agent...")
            train_agent(
                config_path=args.config,
                agent_type=args.agent,
                total_timesteps=args.timesteps,
                output_dir=args.output,
                seed=args.seed
            )
            print("\nTraining complete!")
        
        elif args.command == 'eval':
            print(f"Evaluating {args.agent.upper()} agent...")
            evaluate_agent(
                config_path=args.config,
                model_path=args.model,
                agent_type=args.agent,
                n_episodes=args.episodes,
                output_dir=args.output
            )
            print("\nEvaluation complete!")
        
        elif args.command == 'tune':
            print("Starting hyperparameter tuning...")
            tune_hyperparameters(
                config_path=args.config,
                n_trials=args.trials,
                n_timesteps=args.timesteps,
                output_dir=args.output,
                timeout=args.timeout,
                n_jobs=args.jobs
            )
            print("\nTuning complete!")
        
        elif args.command == 'walkforward':
            print("Starting walk-forward validation...")
            rolling_walkforward(
                config_path=args.config,
                train_window_days=args.train_window,
                test_window_days=args.test_window,
                step_days=args.step,
                agent_type=args.agent,
                timesteps_per_fold=args.timesteps,
                output_dir=args.output
            )
            print("\nWalk-forward validation complete!")
        
        elif args.command == 'report':
            print("Generating report...")
            if args.text_only:
                print_summary(args.results_dir)
            else:
                generate_report(args.results_dir, args.output)
                print_summary(args.results_dir)
            print("\nReport generation complete!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
