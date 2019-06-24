#!/bin/bash

set -eu

# メッセージを一時保存する場所
MESSAGEFILE=$(mktemp -t webhooksXXXX)
# 終了時に削除
trap "rm ${MESSAGEFILE}" 0

# パイプの有無
if [ -p /dev/stdin ]; then
  # 改行コードを変換して格納
  cat - | tr '\n' '\\' | sed 's/\\/\\n/g' > ${MESSAGEFILE}
else
  echo "nothing stdin"
  exit 1
fi

# WebHookのURL
URL='https://hooks.slack.com/services/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'
# 送信先のチャンネル
CHANNEL=${CHANNEL:-'#notify-end-of-process'}
# botの名前
BOTNAME=${BOTNAME:-'Notify'}
# 絵文字
EMOJI=${EMOJI:-':squirrel:'}
# 見出し
HEAD=${HEAD:-"ABCI\n"}

# メッセージをシンタックスハイライト付きで取得
MESSAGE='```'`cat ${MESSAGEFILE}`'```'

# json形式に整形
payload="payload={
  \"channel\": \"${CHANNEL}\",
  \"username\": \"${BOTNAME}\",
  \"icon_emoji\": \"${EMOJI}\",
  \"text\": \"${HEAD}${MESSAGE}\"
}"

# 送信
curl -s -S -X POST --data-urlencode "${payload}" ${URL} > /dev/null
