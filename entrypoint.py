import os
from argparse import ArgumentParser
from subprocess import call

PROJECT_NAME = 'cloud_ml_app'
MAIN_SERVICE_NAME = 'service-app'
# DATA_SOURCE variable could be introduced in order to use it in docker compose
os.environ['SOURCE_DATA'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_store')
os.environ['JUPYTER_NOTEBOOKS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'jupyter_notebooks')

docker_compose = f"""docker-compose --project-name {PROJECT_NAME} -f docker_compose/docker-compose.yml"""
docker_compose_postfix = f" --rm --name {PROJECT_NAME} {MAIN_SERVICE_NAME} "
simple_run = f'{docker_compose} run {docker_compose_postfix}'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--scenario', dest='scenario', required=True, help='work scenario')
    args = parser.parse_args()
    if args.scenario == 'down':
        sh_command = f'{docker_compose} {args.scenario}'
    elif args.scenario == 'jupyter':
        sh_command = f'{docker_compose} run -p 8889:8888 --rm --name {PROJECT_NAME}_jupyter jupyter-app {args.scenario}'
    else:
        raise ValueError('The scenario you specified doesn\'t exist: %s' % args.scenario)
    print(sh_command)
    call(sh_command, shell=True)