import os

class Utils(object):
    def __init__(self): pass

    @classmethod 
    def check_dir(self, file_dir):
        '''
        Check directory exists; if not make it.
        '''
        if not os.path.isdir(file_dir):
            print ('making directory: {}'.format(file_dir))
            os.system('mkdir -p %s' %file_dir)

    @classmethod 
    def sub_hp_script(self, eq_weights, hp_string, k_folds, pt_rew, job_dir='{}/submissions/bdt_hp_opts_jobs'.format(os.getcwd())):
        '''
        Submits train_bdt.py with option -H hp_string -k, to IC batch
        When run this way, a BDT gets trained with HPs = hp_string, and cross validated on k_folds 
        '''

        file_safe_string = hp_string
        for p in [':',',','.']:
            file_safe_string = file_safe_string.replace(p,'_')

        os.system('mkdir -p {}'.format(job_dir))
        sub_file_name = '{}/sub_bdt_hp_{}.sh'.format(job_dir,file_safe_string)
        #FIXME: add config name as a function argument to make it general
        sub_command   = "python train_bdt.py -c bdt_config.yaml -H {} -k {}".format(hp_string, k_folds)
        if eq_weights: sub_command += ' -w'
        if pt_rew: sub_command += ' -P'
        #Change!
        #with open('{}/submissions/sub_bdt_opt_template.sh'.format(os.getcwd())) as f_template:
        with open('{}/submissions/sub_hp_opt_template.sh'.format(os.getcwd())) as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)
        os.system( 'qsub -o {} -e {} -q hep.q -l h_rt=1:00:00 -l h_vmem=4G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )

    @classmethod 
    def sub_lstm_hp_script(self, eq_weights, batch_boost, hp_string, pt_rew, job_dir='{}/submissions/lstm_hp_opts_jobs'.format(os.getcwd())):
        '''
        Submits train_bdt.py with option -H hp_string -k, to IC batch
        When run this way, a LSTM gets trained with HPs = hp_string
        '''

        file_safe_string = hp_string
        for p in [':',',','.']:
            file_safe_string = file_safe_string.replace(p,'_')

        os.system('mkdir -p {}'.format(job_dir))
        sub_file_name = '{}/sub_lstm_hp_{}.sh'.format(job_dir,file_safe_string)
        #FIXME: add config name as a function argument to make it general. Do not need file paths here as copt everything into one dir
        sub_command   = "python train_lstm.py -c lstm_config_ggh.yaml -H {}".format(hp_string)
        if eq_weights: sub_command += ' -w'
        if batch_boost: sub_command += ' -B'
        if pt_rew: sub_command += ' -P'
        with open('{}/submissions/sub_hp_opt_template.sh'.format(os.getcwd())) as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)
        os.system( 'qsub -o {} -e {} -q hep.q -l h_rt=3:00:00 -l h_vmem=12G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )

