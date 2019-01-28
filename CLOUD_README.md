# Instructions to run gridfinder on cloud through SSH

1. Open your terminal and type the following and enter the password.
```
$ ssh <user>@<IP>
```

2. There are several different processes running in different `tmux` sessions. I prepared two sessions called `g1` and `g2` that we can use to run the algorithm in parallel. To connect to one of the sessions, type the below. There should now be an extra bar on the bottom of the terminal with the session name in it.
```
$ tmux attach -t <name>
```

3. Make sure you are working in the correct Python environment. The prompt should start with `(env) gost@...`. If not, type the below and make sure that `(env)` is there now.
```
$ source env/bin/activate
```

4. Run the below, which will print out the details of how to use the command-line script for gridfinder.
```
$ python gridfinder/quickrun.py -h
```

5. An example to do Rwanda with percentile 60 and upsample 2 would be:
```
$ python gridfinder/quickrun.py --country=Rwanda --percentile=60 --upsample=2
```

6. The algorithm is split in three parts: prepare NTL, prepare roads, run model. You can skip one of the first two steps by adding `--skip-ntl` or `--skip-roads` when you run the script. If you change `percentile`, `upsample` or `threshold` you *can't* skip NTL. If you change `upsample` you *can't* skip roads, but skip roads whenever you can because it is *very* slow.

7. Check that the model starts running and displays progress updates. Then you can leave this session (without stopping the model) by typing `Ctrl-B d`. Check that bar on the bottom is gone. 

8. Now you can attach to another `tmux` session, or if you want to check memory use type `free`.

9. To download results, type the following to see what zip files are available.
```
$ l download/
```

10. Then open another terminal on your laptop and type the following to copy it to your laptop:
```
$ scp <user>@<IP>:~/download/<filename> /somewhere/on/local/machine
```

11. Pay attention the next time you SSH in, as it displays the current hard drive space available: `Usage of /: 83.2%...`. We might need to delete some stuff in `output/` if this gets above 95% or something.

12. Make sure that you are in a session(`g1` or `g2`) before running the model, and make sure that you have `(env)` at the beginning of the prompt.
