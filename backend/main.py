import argparse
import sys
import run_trial
import main_mirror

def main():
    # 1. Pre-parse to find --mode
    parser = argparse.ArgumentParser(description="MIRAI Unified Entry Point")
    parser.add_argument("--mode", choices=["api", "local"], required=True, 
                      help="Mode of operation: 'api' for Magic Mirror, 'local' for VTON Trial")
    
    # We use parse_known_args because the sub-scripts might have their own arguments
    # that this top-level parser doesn't know about (especially for 'local' mode).
    args, remaining_argv = parser.parse_known_args()
    
    print(f"=== MIRAI Launcher: Starting in {args.mode.upper()} mode ===")
    
    if args.mode == "api":
        # === API MODE ===
        # Launches the interactive Main Mirror Application
        # This mode typically doesn't take extra CLI args, so we ignore them.
        try:
            main_mirror.main()
        except KeyboardInterrupt:
            print("\nMirror session ended by user.")
            
    elif args.mode == "local":
        # === LOCAL MODE ===
        # Launches the Headless VTON Trial runner
        # We need to pass the remaining arguments to run_trial.py
        # run_trial uses argparse, which reads sys.argv by default.
        # We patch sys.argv to remove '--mode' and its value, so run_trial sees only its own args.
        
        # Construct new argv: [script_name, ...remaining_args]
        # We use sys.argv[0] as the script name
        new_argv = [sys.argv[0]] + remaining_argv
        
        print(f"Passing arguments to VTON runner: {remaining_argv}")
        
        # Patch sys.argv
        sys.argv = new_argv
        
        try:
            run_trial.main()
        except SystemExit as e:
            # Catch exit so we don't crash the launcher if sub-script calls exit()
            if e.code != 0:
                print(f"VTON runner exited with code {e.code}")

if __name__ == "__main__":
    main()
