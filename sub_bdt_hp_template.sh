#!/bin/bash

#inputs to be changed by python wrapper
echo "about to run"
MYDIR=!CWD!
CMD=!CMD!

#execution
cd $MYDIR
eval `scramv1 runtime -sh`
echo "got to cmsenv"
cd $TMPDIR
mkdir -p scratch_$RAND
cd scratch_$RAND
cp -p $MYDIR/*.py .
cp -p $MYDIR/*.yaml .
echo "About to run the following command:"
echo $CMD
if ( $CMD ) then
  touch $NAME.done
  echo 'Success!'
else
  touch $NAME.fail
  echo 'Failure..'
fi
cd -
rm -r scratch_$RAND


 
