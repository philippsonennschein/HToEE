#!/bin/bash

#inputs to be changed by python wrapper
echo "about to run"
MYDIR=!CWD!
CMD=!CMD!

#execution
cd $MYDIR
eval `scramv1 runtime -sh`
cd $TMPDIR
mkdir -p scratch_$RAND
cd scratch_$RAND
cp -p $MYDIR/python/*.py .
cp -p $MYDIR/training/*.py .
cp -p $MYDIR/configs/*.yaml .

echo "DEBUG: dir is::"
echo ${CWD}
echo "DEBUG: files in dir are:"
ls .

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


 
