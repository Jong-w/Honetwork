import gym

# gym 라이브러리에서 모든 환경을 로드합니다.
all_envs = gym.envs.registry.all()

# 환경 이름이 'NoFrameskip-v4'로 끝나고, 'ram'을 포함하지 않는 환경만 필터링합니다.
noframeskip_v4_no_ram_envs = [env.id for env in all_envs if ((env.id.endswith('NoFrameskip-v4')) and ('-ram' not in env.id))]

print(noframeskip_v4_no_ram_envs)
