import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    if current_vel < 11:
      action.acceleration = 1
    else:
      action.acceleration = 0
    action.steer = 0.25
    action.drift = False

    
    # config = pystk.RaceConfig()
    # print(aim_point)
    # action = pystk.Action()
    # action = pystk.Action(10, 0, 1, True)

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    # config = pystk.RaceConfig()
    # k = pystk.Race(config)
    # state = pystk.WorldState()
    # k.start()
    # k.step()
    # state.update()
    # data = []
    # current_action = 
    # try:
    #     for i in range(n_step):
    #         x = torch.as_tensor(np.array(k.render_data[0].image))[None].permute(0,3,1,2).float()/255. - 0.5
    #         # a = actor(x.to(device))[0]
    #         k.step(pystk.Action(steer=float(a[0]), acceleration=float(a[1]), brake=float(a[2])>0.5))
    #         state.update()
    #         data.append( (np.array(k.render_data[0].image), (state.karts[0].distance_down_track)) )
    # finally:
    #     k.stop()
    #     del k

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
