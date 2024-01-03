from modules import launch_utils
import sys

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if '--noprepare' not in sys.argv and not args.skip_prepare_environment:
            prepare_environment()

    if args.test_server:
        configure_for_tests()

    start()


if __name__ == "__main__":
    # sys.argv.append("--nowebui")
    # sys.argv.append("--noprepare")
    # sys.argv.append("--memory-ratio-allow")
    # sys.argv.append("0.1")
    # sys.argv.append("--memory-check-interval")
    # sys.argv.append("1")
    # sys.argv.append("--memory-timeout")
    # sys.argv.append("60")
    # sys.argv.append("--gpu-memory-fraction")
    # sys.argv.append("0.9")
    # sys.argv.append("--xformers")
    # sys.argv.append("--api")
    # sys.argv.append("--port")
    # sys.argv.append("7865")
    # sys.argv.append("--additional-model")
    # sys.argv.append(r"C:\Users\Personify.inc\AppData\Local\Personify\AiGenerator\Models")
    # sys.argv.append("--additional-lora")
    # sys.argv.append(r"C:\Users\Personify.inc\AppData\Local\Personify\AiGenerator\Styles\Lora")
    # sys.argv.append("--additional-embedding")
    # sys.argv.append(r"C:\Users\Personify.inc\AppData\Local\Personify\AiGenerator\Styles\Embeddings")
    main()
