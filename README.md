### Rouest's Project Page



<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<title>ProjectEDA</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.6 (http://getbootstrap.com)
 * Copyright 2011-2015 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.2.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.2.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
@-moz-document url-prefix() {
  div.inner_cell {
    overflow-x: hidden;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 20ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: "(command)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}

@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Part-1-of-Data-822-Project:-Exploratory-Data-Analysis">Part 1 of Data 822 Project: Exploratory Data Analysis<a class="anchor-link" href="#Part-1-of-Data-822-Project:-Exploratory-Data-Analysis">&#182;</a></h1><p>This set of commands lays out the process I used to explore the training dataset for the project. I start by importing our trusty friends pandas, numpy and matplotlib and set some default parameters for matplotlib.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib</span>
<span class="n">matplotlib</span><span class="o">.</span><span class="n">rcParams</span><span class="p">[</span><span class="s1">&#39;figure.figsize&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">)</span>
<span class="n">matplotlib</span><span class="o">.</span><span class="n">pyplot</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">use</span><span class="p">(</span><span class="s1">&#39;ggplot&#39;</span><span class="p">)</span>
<span class="n">pd</span><span class="o">.</span><span class="n">set_option</span><span class="p">(</span><span class="s1">&#39;max.rows&#39;</span><span class="p">,</span> <span class="mi">100</span><span class="p">)</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="30,000-foot-view-of-the-Training-Set">30,000 foot view of the Training Set<a class="anchor-link" href="#30,000-foot-view-of-the-Training-Set">&#182;</a></h2><p>I started by reading in the training set from the project submission repo and taking a look at it.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Let&#39;s read in the training data set</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;~/rouest/project-submissions/data/train.csv&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># And take a basic look. Already we can see a number of object variables that may need to be converted.</span>
<span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 509009 entries, 0 to 509008
Data columns (total 11 columns):
step              509009 non-null int64
type              509009 non-null object
amount            509009 non-null float64
nameOrig          509009 non-null object
oldbalanceOrg     509009 non-null float64
newbalanceOrig    509009 non-null float64
nameDest          509009 non-null object
oldbalanceDest    509009 non-null float64
newbalanceDest    509009 non-null float64
isFraud           509009 non-null int64
id                509009 non-null int64
dtypes: float64(5), int64(3), object(3)
memory usage: 42.7+ MB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>And let's just confirm that we aren't going to need to impute any values or remove any incomplete records.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[7]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>step              0
type              0
amount            0
nameOrig          0
oldbalanceOrg     0
newbalanceOrig    0
nameDest          0
oldbalanceDest    0
newbalanceDest    0
isFraud           0
id                0
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="The-Data-Set:-An-Explainer">The Data Set: An Explainer<a class="anchor-link" href="#The-Data-Set:-An-Explainer">&#182;</a></h2><p>Below I've listed the descriptions of each part of our data set from the project repo listed by index on original read in.</p>
<ol>
<li>step: maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation)</li>
<li>type: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER</li>
<li>amount: amount of the transaction in local currency</li>
<li>nameOrig: customer who started the transaction</li>
<li>oldbalanceOrg: initial balance before the transaction</li>
<li>newbalanceOrig: new balance after the transaction</li>
<li>nameDest: customer who is the recipient of the transaction</li>
<li>oldbalanceDest: initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants)</li>
<li>newbalanceDest: new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants)</li>
<li>isFraud: This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system</li>
<li>id: this is a unique ID number for the transaction record</li>
</ol>
<p>Remember that isFraud at index 9 in our target variable!</p>
<p>Let's see what the data actually looks like:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Looking at the data</span>
<span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">20</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[9]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>303</td>
      <td>CASH_IN</td>
      <td>185164.71</td>
      <td>C1499985475</td>
      <td>3075480.01</td>
      <td>3260644.72</td>
      <td>C1771727877</td>
      <td>881991.88</td>
      <td>696827.18</td>
      <td>0</td>
      <td>540576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>356</td>
      <td>CASH_IN</td>
      <td>79083.65</td>
      <td>C108745493</td>
      <td>5489716.32</td>
      <td>5568799.97</td>
      <td>C1167754301</td>
      <td>153219.51</td>
      <td>74135.86</td>
      <td>0</td>
      <td>120014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>TRANSFER</td>
      <td>2336832.78</td>
      <td>C975415534</td>
      <td>147958.78</td>
      <td>0.00</td>
      <td>C718985478</td>
      <td>5069347.06</td>
      <td>7307970.46</td>
      <td>0</td>
      <td>623141</td>
    </tr>
    <tr>
      <th>3</th>
      <td>238</td>
      <td>TRANSFER</td>
      <td>228517.91</td>
      <td>C1968162743</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>C1544755390</td>
      <td>18768561.09</td>
      <td>18997079.00</td>
      <td>0</td>
      <td>547737</td>
    </tr>
    <tr>
      <th>4</th>
      <td>133</td>
      <td>CASH_IN</td>
      <td>180179.73</td>
      <td>C467196066</td>
      <td>21448.00</td>
      <td>201627.73</td>
      <td>C1386847873</td>
      <td>7160295.13</td>
      <td>6980115.40</td>
      <td>0</td>
      <td>569291</td>
    </tr>
    <tr>
      <th>5</th>
      <td>355</td>
      <td>CASH_OUT</td>
      <td>152809.83</td>
      <td>C835813539</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>C1697188789</td>
      <td>179097.81</td>
      <td>331907.64</td>
      <td>0</td>
      <td>621439</td>
    </tr>
    <tr>
      <th>6</th>
      <td>379</td>
      <td>CASH_OUT</td>
      <td>119283.52</td>
      <td>C1320649033</td>
      <td>320946.00</td>
      <td>201662.48</td>
      <td>C1196142176</td>
      <td>633606.80</td>
      <td>752890.31</td>
      <td>0</td>
      <td>122016</td>
    </tr>
    <tr>
      <th>7</th>
      <td>178</td>
      <td>PAYMENT</td>
      <td>17037.96</td>
      <td>C502419545</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>M1419983756</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>181905</td>
    </tr>
    <tr>
      <th>8</th>
      <td>283</td>
      <td>PAYMENT</td>
      <td>1658.36</td>
      <td>C1804417059</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>M1729883225</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>604829</td>
    </tr>
    <tr>
      <th>9</th>
      <td>233</td>
      <td>DEBIT</td>
      <td>2539.38</td>
      <td>C178740880</td>
      <td>15230.00</td>
      <td>12690.62</td>
      <td>C34403520</td>
      <td>779456.01</td>
      <td>1108570.97</td>
      <td>0</td>
      <td>549197</td>
    </tr>
    <tr>
      <th>10</th>
      <td>42</td>
      <td>CASH_IN</td>
      <td>302335.54</td>
      <td>C443150216</td>
      <td>10528719.85</td>
      <td>10831055.39</td>
      <td>C303009903</td>
      <td>820532.75</td>
      <td>518197.21</td>
      <td>0</td>
      <td>343693</td>
    </tr>
    <tr>
      <th>11</th>
      <td>397</td>
      <td>PAYMENT</td>
      <td>8718.25</td>
      <td>C974457792</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>M1324377813</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>634364</td>
    </tr>
    <tr>
      <th>12</th>
      <td>185</td>
      <td>CASH_IN</td>
      <td>118441.00</td>
      <td>C2081805049</td>
      <td>102465.00</td>
      <td>220906.00</td>
      <td>C1012713660</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>48412</td>
    </tr>
    <tr>
      <th>13</th>
      <td>351</td>
      <td>CASH_OUT</td>
      <td>186696.51</td>
      <td>C1046187494</td>
      <td>134471.00</td>
      <td>0.00</td>
      <td>C205492948</td>
      <td>2852.24</td>
      <td>189548.75</td>
      <td>0</td>
      <td>416717</td>
    </tr>
    <tr>
      <th>14</th>
      <td>11</td>
      <td>PAYMENT</td>
      <td>6113.37</td>
      <td>C1673645214</td>
      <td>105.00</td>
      <td>0.00</td>
      <td>M52236598</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>543932</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>CASH_OUT</td>
      <td>39297.65</td>
      <td>C869501792</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>C1758687081</td>
      <td>152234.34</td>
      <td>717320.94</td>
      <td>0</td>
      <td>143067</td>
    </tr>
    <tr>
      <th>16</th>
      <td>232</td>
      <td>CASH_IN</td>
      <td>64653.54</td>
      <td>C1558381904</td>
      <td>1746874.41</td>
      <td>1811527.95</td>
      <td>C1843550991</td>
      <td>962760.52</td>
      <td>898106.98</td>
      <td>0</td>
      <td>571091</td>
    </tr>
    <tr>
      <th>17</th>
      <td>227</td>
      <td>CASH_OUT</td>
      <td>119832.17</td>
      <td>C1488033658</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>C1008168506</td>
      <td>191039.29</td>
      <td>310871.46</td>
      <td>0</td>
      <td>145708</td>
    </tr>
    <tr>
      <th>18</th>
      <td>202</td>
      <td>CASH_OUT</td>
      <td>380870.26</td>
      <td>C389555387</td>
      <td>361.00</td>
      <td>0.00</td>
      <td>C1143935957</td>
      <td>136647.16</td>
      <td>517517.42</td>
      <td>0</td>
      <td>417571</td>
    </tr>
    <tr>
      <th>19</th>
      <td>307</td>
      <td>CASH_OUT</td>
      <td>298693.47</td>
      <td>C669123051</td>
      <td>163778.00</td>
      <td>0.00</td>
      <td>C163613986</td>
      <td>0.00</td>
      <td>298693.47</td>
      <td>0</td>
      <td>57840</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Since I don&#39;t see any fraudulent records in the firs head() call, let&#39;s make sure they are there!</span>
<span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[12]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>636</th>
      <td>265</td>
      <td>CASH_OUT</td>
      <td>881187.51</td>
      <td>C1389553281</td>
      <td>881187.51</td>
      <td>0.00</td>
      <td>C372435481</td>
      <td>35321.01</td>
      <td>916508.52</td>
      <td>1</td>
      <td>163801</td>
    </tr>
    <tr>
      <th>881</th>
      <td>190</td>
      <td>CASH_OUT</td>
      <td>605029.34</td>
      <td>C1568238028</td>
      <td>605029.34</td>
      <td>0.00</td>
      <td>C506455454</td>
      <td>211936.17</td>
      <td>816965.52</td>
      <td>1</td>
      <td>88860</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>160</td>
      <td>TRANSFER</td>
      <td>548161.93</td>
      <td>C814345778</td>
      <td>548161.93</td>
      <td>0.00</td>
      <td>C1464577173</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>507125</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>425</td>
      <td>CASH_OUT</td>
      <td>486175.40</td>
      <td>C1737680610</td>
      <td>486175.40</td>
      <td>0.00</td>
      <td>C1596545373</td>
      <td>0.00</td>
      <td>486175.40</td>
      <td>1</td>
      <td>90109</td>
    </tr>
    <tr>
      <th>6406</th>
      <td>451</td>
      <td>TRANSFER</td>
      <td>9265.43</td>
      <td>C1329487815</td>
      <td>9265.43</td>
      <td>0.00</td>
      <td>C1936702457</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>463575</td>
    </tr>
    <tr>
      <th>6419</th>
      <td>186</td>
      <td>TRANSFER</td>
      <td>179321.12</td>
      <td>C1132083905</td>
      <td>179321.12</td>
      <td>0.00</td>
      <td>C955198432</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>53504</td>
    </tr>
    <tr>
      <th>7871</th>
      <td>514</td>
      <td>TRANSFER</td>
      <td>10880.26</td>
      <td>C132837208</td>
      <td>10880.26</td>
      <td>0.00</td>
      <td>C457634395</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>184398</td>
    </tr>
    <tr>
      <th>9953</th>
      <td>161</td>
      <td>TRANSFER</td>
      <td>10000000.00</td>
      <td>C831325954</td>
      <td>11336901.11</td>
      <td>1336901.11</td>
      <td>C1166671647</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>129097</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>217</td>
      <td>TRANSFER</td>
      <td>972713.05</td>
      <td>C706587857</td>
      <td>972713.05</td>
      <td>0.00</td>
      <td>C1000855680</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>179789</td>
    </tr>
    <tr>
      <th>12400</th>
      <td>724</td>
      <td>CASH_OUT</td>
      <td>72389.42</td>
      <td>C2090432901</td>
      <td>72389.42</td>
      <td>0.00</td>
      <td>C2044300505</td>
      <td>397599.55</td>
      <td>469988.97</td>
      <td>1</td>
      <td>113267</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Let&#39;s also take a look at isFraud = 1 as a percentage of our overall data set</span>
<span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 669 entries, 636 to 507597
Data columns (total 11 columns):
step              669 non-null int64
type              669 non-null object
amount            669 non-null float64
nameOrig          669 non-null object
oldbalanceOrg     669 non-null float64
newbalanceOrig    669 non-null float64
nameDest          669 non-null object
oldbalanceDest    669 non-null float64
newbalanceDest    669 non-null float64
isFraud           669 non-null int64
id                669 non-null int64
dtypes: float64(5), int64(3), object(3)
memory usage: 62.7+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">&#39;The percentage of our dataset that has isFraud = 1 is </span><span class="si">%s</span><span class="s1"> percent.&#39;</span> <span class="o">%</span> <span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">/</span><span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">*</span><span class="mi">100</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>The percentage of our dataset that has isFraud = 1 is 0.13143186073330726 percent.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Working-through-the-disctinction-between-isFraud-and-NOT-isFraud">Working through the disctinction between isFraud and NOT isFraud<a class="anchor-link" href="#Working-through-the-disctinction-between-isFraud-and-NOT-isFraud">&#182;</a></h2><p>In the following section, I am going to try to work methodically through each of our features to see any obvious deliniations for our taget variable. Let's start with step:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># So we don&#39;t have to include the logic on each call of df, let&#39;s make an isFraud = 1 df called dfraud</span>
<span class="n">dfraud</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Looking-at-the-whole-set--vs.-the-isFraud-set-by-step">Looking at the whole set  vs. the isFraud set by step<a class="anchor-link" href="#Looking-at-the-whole-set--vs.-the-isFraud-set-by-step">&#182;</a></h2><p>Below I wanted to compare the a normed histogram for df and dfraud to see any comparative differences. I used normed = True in order to overlay them and actually be able to see the isFraud set.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">steps</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;All Records&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">df</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">ifsteps</span> <span class="o">=</span> <span class="n">dfraud</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">steps</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">ifsteps</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkcAAAFqCAYAAAAQmf6CAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAPYQAAD2EBqD+naQAAIABJREFUeJzs3XmcFPW1//9X98yAg2FxBCEGI4pGI8Y1iaCCJCwKXiN6
9RjUaHCLihpxQbzXnyMkblyBGCO/GDcwRm7OvRo0KuCuQbmJuwlGDQqKJuKCAuqwzEx//6gaLJru
YXqqZrpn5v18PHhIf+rUqc+np9s5fKrqU6lMJoOIiIiIBNLF7oCIiIhIKVFxJCIiIhKh4khEREQk
QsWRiIiISISKIxEREZEIFUciIiIiESqORERERCJUHImIiIhEqDgSERERiVBxJCIiIhJRXuwO5GNm
44GLgD7Ay8C57v5sI/FDgWnAAOAd4Ep3n50VcywwBegHvAFMcvd5ke1nAmeF2wEWA1PcfX4k5nbg
5KzDz3f30c0Y41h3n1Pofm2Nxtm+aJztS0cZJ3ScsWqc8ZXkzJGZHUdQ6FQD+xIURwvMrGee+H7A
/cCjwN7A9cAtZjYiEnMgcBdwM7APcC8w18z2iKRaDlwC7AfsDzwG3Gtm38w65DygN0Hh1gcY28yh
Nne/tkbjbF80zvalo4wTOs5YNc6YSnXmaAJwk7vfARtndA4HTgGm5og/C3jL3SeGr183s4PDPA+H
becB89x9evj68rB4Ogc4G8DdH8jKe5mZnQUMBP4eaV/n7h/GGaCIiIiUppIrjsysgmDW5qqGNnfP
mNkjwKA8uw0EHslqWwDMiLweRDAblR1zZJ5+pAEDugCLsjYPNbMVwCcEs0uXufvKfGMSERGRtqMU
T6v1BMqAFVntKwhOYeXSJ098NzPrvIWYTXKa2Z5mtgZYB8wEjnL31yIh84CTgO8DE4FDgAfNLLWF
cYmIiEgbUHIzRyXgNYLrlroDxwB3mNmQhgLJ3T0Su9jM/gq8CQwFHi/gONuOGjXqa8CBwNokOl6q
BgwY0J3gOq52TeNsXzTO9qejjLWDjHOr8HfotsDHSScvxeLoI6CO4ILnqN7A+3n2eT9P/Gp3X7eF
mE1yunst8Fb48kUz+y7wU4Lrmjbj7kvN7CNgF/IUR2Y2lqwLx0aNGvW1cePG7Qc8nWdM7UZ1dTXA
88XuR0vTONsXjbP96Shj7SjjHDduHLfffvtD8+bNey9r05y4d7GVXHHk7hvM7HlgGHAfQHjKahjw
yzy7LQJGZbWNZNNrhRblyDGCza8nypYGOufbaGZ9CSrXf+WLCX9I2T+oA4GnP/nkE2pra7fQhbat
W7durF69utjdaHEaZ/uicbY/HWWsHWGc5eXlbLPNNowbN+7ccePGPZN4/qQTJmQ6MCsskv5CcNdZ
F2AWgJldDWzv7g3rDf0aGG9m1wK3ERRBxwDRtYeuB54wswuABwhmcvYHTm8IMLOrCK4pegfoCpxA
cE3RyHD71gTLC9xNMOO0C3AtwZpJCwoc41qA2tpaNmzYUOCubUsmk2n3YwSNs73RONufjjLWjjLO
UItcllKKF2Q3XNdzEcGCjS8CewGHRm6f7wPsEIlfRnCr/3DgJYJi6lR3fyQSswg4HjgjjDkaONLd
X40cejtgNsF1R48QFE8j3f2xcHtd2Jd7gdcJ1kx6Fhji7h3mkygiItKepTKZTLH70FHtBzz/4Ycf
tvsKv6qqipUr2/9KBxpn+6Jxtj8dZawdYZwVFRX06tULgkmMF5LOX5IzRyIiIiLFouJIREREJELF
kYiIiEhEqd6tJiIiJapHjx6k023v39bpdJqqqqpid6PFtYdx1tfX8+mnnxbt+CqORESkIOl0ut1f
8CvFVeziTsWRiDRJ+eer4bM11Hy8gvIkFy79Sldqt+6WXD4RkZhUHIlI03y2hnU3T6OuvIza2rrE
0nY+/UJQcSQiJaTtnTQWERERaUEqjkREREQiVByJiIiIRKg4EhERaaa+ffsyY8aMja9///vf07dv
X957770i9qp4zj//fAYOHFjsbsSm4khERCSHWbNm0bdvX4444ogm75NKpUilUluMmz59On379t34
p1+/fgwcOJDLL7+c1atXx+l2UTV1/KVOd6uJiEhiGpZ8KKqEloeYO3cuX//613nppZd4++232XHH
HRPo3JdSqRTXXHMNXbp04YsvvmDhwoXcdttt/O1vf+Oee+5J9FhSGBVHIiKSnHDJh2JKYnmId955
h+eee45bb72ViRMncs899zBhwoSEevil0aNHs8022wBwwgknkEql+OMf/8jLL7/M3nvvnfjx4qqp
qaGysrLY3WhxOq0mIiKS5Z577qFHjx4MGzaMww8/nD/84Q+tctzvfve7ACxbtmyzbXfffTejRo2i
f//+DBgwgLPPPpt//vOfm8W98MIL/OhHP2LAgAHsuuuuDB8+nFtvvXWTmIULF3LUUUex6667ssce
e3DKKaewZMmSTWKmTZtG3759+cc//sH48eMZMGAARx111Mbt8+fP5/vf/z79+/dn+PDhzJ8/P+eY
7r33XkaNGsVuu+3G7rvvnrM/pUbFkYiISJa5c+cyevRoysvLGTNmDEuXLuWVV15p8eMuX74cCJ5f
F3X99ddz/vnn079/f6644gpOP/10Fi5cyDHHHMOaNV+exnzqqac45phjWLJkCaeddhrV1dUcdNBB
PProo5vEnHjiiaxcuZILL7yQM844g+eee44xY8ZsciF5w7VDP/nJT1i3bh2TJk3i+OOPB+DJJ5/k
jDPOoKysjEsvvZRDDz2UCy64YLP36KmnnmL8+PFss802/Od//if/8R//wYEHHshzzz2X7BuXMJ1W
ExERiXjllVdYsmQJV155JRDM5vTp04d77rmHvfbaK9FjffLJJ2QyGWpqali4cCGzZ8+mZ8+eHHDA
ARtj3nvvPaZPn86kSZMYP378xvbRo0czcuRIZs+ezTnnnEN9fT2XXHIJffr04aGHHuIrX/lKzmP+
/Oc/Z5tttuGPf/wj3boFpx8PPfRQDj30UK677rpN7r4DGDBgADfccMMmbVdeeSW9evVi7ty5bL31
1gAMHDiQsWPHssMOO2yMe/TRR+nWrRt33XVXvDeqlWnmSEREJOKee+5hu+2248ADD9zY9oMf/ID7
7ruPTCaT2HEymQxDhgxhr7324oADDuDCCy9kp5124re//S1bbbXVxrgHHniATCbDv/3bv7Fy5cqN
f3r27MlOO+3EM888A8Bf//pXli9fzmmnnZa3MPrggw949dVXMbONhRHAN7/5TYYMGcJjjz22SXwq
leJHP/pR3hwNhRHA4MGD+cY3vrFJbPfu3fniiy944oknmvUeFYtmjkREREL19fX88Y9/5MADD+Tt
t9/e2L7PPvtw00038ac//YkhQ4YkcqxUKsUtt9zC1ltvzccff8xtt93G8uXLNymMILj+qL6+noMO
OihnjoqKCgDefvttUqnUZgVK1LvvvgvAzjvvvNm2XXbZhSeffHKzi66jM0HRHP369dssR//+/fnb
3/628fXJJ5/M/fffz49+9CN69+7NIYccwhFHHMHQoUPz9rEUqDgSEREJLVy4kBUrVnDvvfcyd+7c
TbalUin+8Ic/JFYcQXDKruFutREjRjBs2DDOOeecTS5urq+vJ51Oc+edd5JOb37CJzp70xKyi7VC
bLvttjz00EM88cQTPP744zz++OP8/ve/59hjj93s9F0pUXEkIiISuueee+jVqxdXXXXVZqfQHnzw
QebPn88111xD586dEz92ly5dmDBhAhdeeCH33XcfP/jBD4BghiaTybDDDjuw00475d2/Ie7111/n
4IMPzhnTt29fAN56663Ntr355ptUVVVt8Vb9hhxLly7NmSNbeXk5w4cPZ/jw4QBMmjSJ3/3ud5x/
/vmJrx2VFF1zJCIiAqxdu5b58+czYsQIRo0axejRozf58+Mf/5g1a9bw0EMPtVgfjj76aPr06cPM
mTM3to0aNYp0Os306dNz7vPJJ58A8K1vfYuvf/3r3HLLLXlX2d5uu+0YMGAA//M//7PJXW6vvfYa
Tz75JMOGDdtiH6M5Pvvss43tTz31FG+88UbOvkXtvvvuAKxfv36LxyoWzRyJSLuTxCrNNR+voLy2
dtPGhFZeltK0YMECPvvsM0aMGJFz+/7778+2227LH/7wh4IeKVKI8vJyTj31VH7+85/z5JNPcsgh
h7DjjjsyceJErrnmGpYvX85hhx3G1ltvzTvvvMP8+fM58cQT+clPfkIqleLqq69m3LhxjBw5kuOO
O47tttuOJUuW8I9//IM777wTgMsuu4yTTjqJI444gh/+8IfU1NQwa9YsunfvzgUXXNCkfl566aWc
fPLJjBkzhuOOO45PPvmEWbNmsfvuu/P5559vjLv44ov59NNPOeigg/jqV7/K8uXLmTVrFnvuuSe7
7rpri7yHSVBxJCLtTwKrNNeVl1FbW7dJWxIrL0vpmjt3LpWVlQwePDjn9lQqxbBhw5g7dy6ffvop
PXr0aJHniJ144onccMMN/OpXv+KQQw4BYPz48fTv35+bb75547U622+/Pd/73vcYOXLkxn0POeQQ
3J0ZM2bwm9/8hvr6enbccUdOOOGEjTGDBw/mzjvvZNq0aUybNo2KigoGDRrEpZdeuvGU2ZYMHTqU
m266ialTp3LNNdfQr18/ZsyYwfz58/m///u/jXH//u//zu9+9zvuuOMOVq9eTa9evTjyyCObXIQV
SyrJ2xKlIPsBz3/44Yds2LCh2H1pUVVVVaxcubLY3Whx7X2c5SveY93N0yjPUTTE0fn0C6nt/bXE
8sGXfY2VI09xlHRfi605n9vG9mlPz1aT4tnS57KiooJevXoB7A+8kPTxNXMkIiKJqd26m2bXpM3T
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
rkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Looking at the records with step &gt; 400, it almost appears as those isFraud follows the same distribution; let's take a closer look.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># We&#39;ll start with a non-fraud dataframe</span>
<span class="n">dfnot</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># And then run the same plot but with the range set for dfraud.step.max()</span>
<span class="n">steps2</span> <span class="o">=</span> <span class="n">dfnot</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">dfnot</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">dfraud</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">ifsteps2</span> <span class="o">=</span> <span class="n">dfraud</span><span class="o">.</span><span class="n">step</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">steps2</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">ifsteps2</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
7rvvzkknncS999673dmjiy++mPvuu4+vf/3rfPe73+Wll17ixhtv5M033+TWW28FYPbs2fzgBz9g
p5124qKLLiKTyTB06NDt9qO5uXmbx7TstNNO9OvXb+vrLVu2cNppp3HIIYfQ0NDAgAEDAPiP//gP
Nm/eTH19PUOGDOHFF1/k9ttv5/333+9Q9KVSqbzHj27LZDJ85zvf4aWXXuKMM85g9OjRPPLII8yc
ObPTHOVCxZGIiEhMM2bM4L777uOmm25i1qxZOWNeffVV7rvvPk477TTmzJkDwOmnn87OO+/MLbfc
wrJlyxg/fjxHHXUUc+bMoa6ujuOPP77LfVi0aBH33nvv1tepVIprr722w6zTpk2b+MY3vsHFF1/c
Yd+Ghgb69++/9fWpp57Kbrvtxrx58/jhD3/I8OHDu9wPgEceeYTf//73zJo1i7POOmvrWE888cSC
8pSKTquJiIjEtPvuu/ONb3yDe+65hzVr1uSMeeqpp0ilUpxzzjkd2r/73e+SyWR48sknY/Xh6KOP
5pe//OXWP4sWLWLChAnbxH3729/epi1aGDU1NbF27Vq+8IUvkMlkWLFiRcF9+fWvf03//v057bTT
tral02nq6+vJZDIF5+tpmjkSERFJwEUXXcTixYu58cYbc84erVq1inQ6zZ577tmhfdiwYQwePJhV
q1bFOv4nP/lJDjvssE5j+vXrl/O036pVq5g7dy5PPvkk69at29qeSqXYsKHwOwpXrVrFiBEjtrlb
bq+99io4VymoOBIREUnA7rvvzoknnsg999zDtGnT8saV8pqbXLf2t7a2csopp7Bx40amT5/OXnvt
RW1tLe+99x4XX3wxbW1tW2Pz9T0aUwl0Wk1ERCQhF110ES0tLdx0003bbBs5ciRtbW289dZbHdo/
+OAD1q1bx8iRI7e29WQBtWLFCt555x1mzZrFeeedx+TJkznssMNyzjANHjyY9evXb9OePes1cuRI
Vq9ezaZNmzq0v/HGG8l2vkhUHImIiCRkjz324MQTT+Tuu+/e5tqjr371q2QyGW677bYO7bfccgup
VIqJEydubautrc1ZhBRDOh2UAtFrgTKZDLfffvs2Rdoee+zBa6+9xkcffbS17Y9//CMvvvhih7iv
fvWrNDc3c/fdd29ta21t5c4779TdaiIipZDEystNf2+kuqWlY2PMlZSl8uS6uHjGjBksXryYN998
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
tlQqxV/+8petr99++21+/etfs+eee3aI++1vf9vh9Np3vvMdJk+ezK233tqt4mju3LlUVVVx//33
s8suuwBwzDHHMGnSpLK63KLclWVxBODu8wnuFsu1rT5H2zMEM0Gd5VwMLO5k+9ld7NuPgB91JVZ6
XvXG9fDxhuQT7zSQFj3pXkSypFIprr766m0Kn6jDDjss5/ZoYbRu3TpaW1v54he/yNKlSwvuR0tL
C88++yxf//rXtxZGAJ/5zGc4/PDDee655wrO2VeVbXEk0m0fb6D51nmJp+1/zsWg4khEcjjggAM6
vSB75MiROdsfe+wxfv7zn/OnP/2J5ubmre39+hX+0IU1a9bQ3NzMqFGjttm21157qTgqgIojERGR
Itthhx22aXvuuec466yzOPTQQ7nmmmvYZZddqK6u5t577+WRRx7ZGpdvfaLW1tLf6VepVByJiIiU
wJIlSxgwYAD33HMPVVX/WDbi7rvv7hA3ePBgIDjtFrVq1aoOr4cNG0b//v15++23tznWG2/o2eiF
0NVZIiIiJZBOp0mn0x1mgN555x0ef/zxDnFDhgxh8ODB/O53v+vQvmDBgg6zStXV1Rx++OEsWbKE
xsZ/LMX35z//mWeffbZIo6hMmjkSERGJobvryk2aNIk77riDU089leOPP57333+fhQsXstdee/H6
6693iJ06dSo333wzAwcO5POf/zzLli3jnXfe2ebYl1xyCccddxzHH388p59+Os3NzSxYsIB99tln
m5ySn2aOREREYtjeM8vyPdfsiCOO4F/+5V9obGykoaGBhx56iCuvvJJJkyZtE3vxxRfzzW9+k4ce
eoirr76aqqoqFi5cuE3uz33uc9x999184hOf4Kc//Sn33Xcfl19+ec6ckp9mjkREpCQG9aviiNFD
St0NBvXr/qNmzAwzy7u9qqqqw3pH2aZOncrUqVO3ab/00ks7vK6treWnP/0pP/3pTzu058o9fvz4
Dhd058sp+ak4EhGRkhjYL120x3aIxKGfShEREZEIFUciIiIiESqORERERCJUHImIiIhEqDgSERER
iVBxJCIiIhKh4khEREQkQsWRiIiISISKIxEREZEIrZAtIiJF1dbWRl1dXam7QTqdpq2trdTdKLpK
GGep+6/iSEREiuqjjz4qdRcAqKurY+3ataXuRtH1lXEWk06riYiIiESoOBIRERGJUHEkIiIiEqHi
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Dang, I half expected all records with steps &gt; 400 to be accounted for by isFraud records. Of course it wasn't going to be that easy. Onward and upward!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Let's-get-into-our-types!">Let's get into our types!<a class="anchor-link" href="#Let's-get-into-our-types!">&#182;</a></h2><p>Below we are going to explore the distribution of our transaction types.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Starting with the whole set</span>
<span class="n">df</span><span class="p">[</span><span class="s1">&#39;type&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[45]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>CASH_OUT    178951
PAYMENT     172717
CASH_IN     111816
TRANSFER     42211
DEBIT         3314
Name: type, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># and then isFraud</span>
<span class="n">dfraud</span><span class="p">[</span><span class="s1">&#39;type&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[47]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>CASH_OUT    339
TRANSFER    330
Name: type, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Interesting! This tells us with some confidence that isFraud = 1 will likely be isolated to those transaction types.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Into-the-floats-we-go,-starting-with-amount!">Into the floats we go, starting with amount!<a class="anchor-link" href="#Into-the-floats-we-go,-starting-with-amount!">&#182;</a></h2><p>Below we are going to start working through our floats including, amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest and newbalanceDest. We aren't including ID as that is not going to be in our feature space.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[72]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">amount</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[72]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    5.090090e+05
mean     1.781422e+05
std      6.030880e+05
min      0.000000e+00
25%      1.336686e+04
50%      7.452952e+04
75%      2.084647e+05
max      9.244552e+07
Name: amount, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[75]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dfraud</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">amount</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[75]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    6.690000e+02
mean     1.434819e+06
std      2.352313e+06
min      0.000000e+00
25%      1.191438e+05
50%      4.475892e+05
75%      1.388952e+06
max      1.000000e+07
Name: amount, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see a significant difference in mean between df a dfraud amounts but let's try to visualize them.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[83]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">amount</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkkAAAF8CAYAAADb3y+3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAPYQAAD2EBqD+naQAAIABJREFUeJzt3XuYXVWZ7/tvkVTCpUmgJFRA8UGPbsQ02ISr7ZaNhxY6
ap/GRl+5eSlgq4DAji0NKloE3CKyBQ5EW1QEYW9ivy3HQ4tiEFrEAzQ0gdYYIG5oFBETxERi063k
UuePOQsnyxmgaq2aa1Xx/TwPT1JzvmuOsUaF1C9jjjlW38jICJIkSXqmLbrdAUmSpF5kSJIkSaph
SJIkSaphSJIkSaphSJIkSaphSJIkSaphSJIkSaphSJIkSaphSJIkSaphSJIkSaoxvdsdqBMROwPn
AQuArYH/DQxl5t2VmrOB44HtgFuBEzLzgcr5mcAFwDuAmcBS4MTMfKxSsz2wGHgLsAm4Bjg1M5+s
1OwCfB44CPgNcCVwRmZuqtTsWV5nX+AxYHFmnj+O931kZi4Z6+s0fo558xzz5jnmzXPMmzcRY95z
M0kRMRp6fgccCuwO/DWwtlJzOvAB4L3AfsCTwNKImFG51EXAm4HDgQOBnSlCUNXV5fUPLmsPBC6t
tLMF8C2KMHkA8G7gPcDZlZptKQLYQ8B84DTgrIg4fhxv/8hxvEbtccyb55g3zzFvnmPevI6PeS/O
JJ0BPJyZ1ZDx05aaU4FzMvM6gIh4F7AaOAzIiJgFHAsckZnfK2uGgPsiYr/MvDMidqcIYXtn5j1l
zcnANyPiQ5m5qjz/KuANmfk4sDwiPgZ8KiLOyswNwDFAP3Bc+fV9EbEX8EHgS50eHEmS1Iyem0kC
/gK4KyIyIlZHxN3VWZmIeBkwF7hp9FhmrgPuAF5bHtqHIgBWa1YCD1dqDgDWjgak0o3ACLB/pWZ5
GZBGLQVmA/MqNbeUAalas1tEzB7rm5ckSb2hF0PSy4ETgJXAIcDfAhdHxDvL83MpgszqltetLs8B
DAJPleFpczVzKdYPPS0zNwJrWmrq2mGMNZIkaZLpxdttWwB3ZubHyq9/EBF/DLwfuKp73ZpwL1qw
YMGLgT8FftvtzrxQzJs3bzbFWjI1xDFvnmPePMe8cVuWP0NfBPyqUxftxZD0C+C+lmP3AX9V/n4V
0EcxW1SdwRkE7qnUzIiIWS2zSYPludGaHauNRMQ0YKClZt+WvgxWzo3+OvgcNc8QEUfSssBswYIF
Lx4aGppPsWhdDRkeHgZY1u1+vJA45s1zzJvnmDdvaGiIyy+//Ibrr7/+5y2nloz3qbdeDEm3Aru1
HNuNcvF2Zj4UEasonkj7IUC5UHt/4LNl/TJgQ1nz9bJmN+ClwO1lze3AdhGxV2Vd0sEUAeyOSs1H
ImKHyrqkQ4AngHsrNZ+IiGnl7brRmpWZ+UTdGyy/Wa3fsD8Fbl27di0bNmyoeZUmwqxZs1i3rvWu
rCaSY948x7x5jnmzpk+fzvbbb8/Q0NDJQ0NDt3Xsup26UAddCNwaER8GkiL8HA/810rNRcCZEfEA
8BPgHOAR4FooFnJHxGXABRGxlmJ/o4uBWzPzzrLm/ohYCnwxIk4AZgCXUCTO0RmgGyjC0FXltgM7
lW0tzsz1Zc3VwMeBL0fEecAewCkUT+CNxW8BNmzYwPr165+rVh0yMjLieDfMMW+eY948x7xrOrpc
pecWbmfmXcBbKW5HLQc+SrHB41crNZ+mCDSXUsz6bAUsyMynKpdaCFwHfA24GXiUYs+kqqOA+yme
arsOuAV4X6WdTRQbTW4EbqPYSPIKYLhSs45i5mhX4C7gfOCszLxsvGMgSZK6r29kZKTbfVBhPrDs
l7/8pf/6aNDAwABr1qzpdjdeUBzz5jnmzXPMm9Xf38+cOXMA9gbufo7y563nZpIkSZJ6gSFJkiSp
hiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJ
kiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSp
hiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJ
kiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSphiFJkiSp
hiFJkiSpxvRud6BVRAwDwy2H78/MV1dqzgaOB7YDbgVOyMwHKudnAhcA7wBmAkuBEzPzsUrN9sBi
4C3AJuAa4NTMfLJSswvweeAg4DfAlcAZmbmpUrNneZ19gceAxZl5/njffx+TI7lueu4SSZImtZ4L
SaUfAQdTZAaADaMnIuJ04APAu4CfAJ8AlkbE7pn5VFl2EbAAOBxYB3yWIgS9vtLG1cBg2c4M4Arg
UuCYsp0tgG8BjwIHADsDVwFPAWeWNdtSBLAbgPcBewCXR8TazPzSeN74yDVfYdMvHhnPSxvTN3Mm
/W8fYv02s7rdFUmSJkyvhqQNmfnLzZw7FTgnM68DiIh3AauBw4CMiFnAscARmfm9smYIuC8i9svM
OyNid+BQYO/MvKesORn4ZkR8KDNXledfBbwhMx8HlkfEx4BPRcRZmbmBIlD1A8eVX98XEXsBHwTG
FZI23nsPGx9cOZ6XNmerrZn+9qFu90KSpAnVq3d2XhkRP4+IByPif5a3vYiIlwFzgZtGCzNzHXAH
8Nry0D4U4a9asxJ4uFJzALB2NCCVbgRGgP0rNcvLgDRqKTAbmFepuaUMSNWa3SJi9rjeuSRJ6gm9
GJL+CXgPxUzO+4GXAbdExDYUAWmEYuaoanV5DopbaE+V4WlzNXMp1g89LTM3AmtaauraYYw1kiRp
Euq5222ZubTy5Y8i4k7gp0AA93enV50VEUcCR1aPzZs3b/bwcOt69d41fXo/2w4MdLsbbevv72dg
CryPycQxb55j3jzHvFl9fcUS5kWLFl24YsWKJ1pOL8nMJeO5bs+FpFaZ+URE/Bh4BXAzxWLuQZ45
gzMIjN46WwXMiIhZLbNJg+W50Zodq+1ExDRgoKVm35buDFbOjf46+Bw1de9pCdD6DZsPLNvca3rN
hg3rWbNmTbe70baBgYEp8T4mE8e8eY558xzzZvX39zNnzhyGh4cXAnd36rq9eLvtGSLijygC0qOZ
+RBF+Di4cn4WxTqi28pDyyiehqvW7Aa8FLi9PHQ7sF25yHrU6NN0d1Rq9oiIHSo1hwBPAPdWag4s
A1a1ZmVmtiZZSZI0ifTcTFJEnA98g+IW24uBRcB64KtlyUXAmRHxAMUWAOcAjwDXQrGQOyIuAy6I
iLUU+xtdDNyamXeWNfdHxFLgixFxAsUWAJdQTMmNzgDdQBGGriq3HdipbGtxZq4va64GPg58OSLO
o9gC4BSKJ/AkSdIk1oszSS+hCB/3UwSjXwIHZOavADLz0xSB5lKKWZ+tgAWVPZIAFgLXAV+juEX3
KMWeSVVHlW3cWNbeQrHXEWU7myg2mtxIMUt1JcVeSsOVmnUUM0e7AncB5wNnZeZl7QyAJEnqvr6R
kZFu90GF+cCyVacczfpJsE/Slud+YUpsJum6geY55s1zzJvnmDdrdE0SsDcvpDVJkiRJ3WBIkiRJ
qmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFI
kiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJ
qmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFI
kiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJqmFIkiRJ
qjG92x14LhFxBvBJ4KLM/GDl+NnA8cB2wK3ACZn5QOX8TOAC4B3ATGApcGJmPlap2R5YDLwF2ARc
A5yamU9WanYBPg8cBPwGuBI4IzM3VWr2LK+zL/AYsDgzz+/cKEiSpKb19ExSROwLvBf4Qcvx04EP
lOf2A54ElkbEjErZRcCbgcOBA4GdKUJQ1dXA7sDBZe2BwKWVdrYAvkURJg8A3g28Bzi7UrMtRQB7
CJgPnAacFRHHj/uNS5KkruvZkBQRfwT8T4rZol+3nD4VOCczr8vMHwHvoghBh5WvnQUcCyzMzO9l
5j3AEPC6iNivrNkdOBQ4LjPvyszbgJOBIyJibtnOocCrgKMzc3lmLgU+BpwUEaOzcMcA/eV17svM
BC4Gnp71kiRJk0/PhiTgs8A3MvMfqwcj4mXAXOCm0WOZuQ64A3hteWgfitmfas1K4OFKzQHA2jJA
jboRGAH2r9Qsz8zHKzVLgdnAvErNLZm5oaVmt4iYPZY3LEmSekdPhqSIOAL4E+DDNafnUgSZ1S3H
V5fnAAaBp8rwtLmauRTrh56WmRuBNS01de0wxhpJkjTJ9NzC7Yh4CcV6oj/LzPXd7o8kSXph6rmQ
BOwNzAHujoi+8tg04MCI+ADFGqE+itmi6gzOIDB662wVMCMiZrXMJg2W50Zrdqw2HBHTgIGWmn1b
+jdYOTf66+Bz1DxDRBwJHFk9Nm/evNnDw8N15T1p+vR+th0Y6HY32tbf38/AFHgfk4lj3jzHvHmO
ebP6+oq4sGjRogtXrFjxRMvpJZm5ZDzX7cWQdCOwR8uxK4D7gE9l5r9GxCqKJ9J+CE8v1N6fYh0T
wDJgQ1nz9bJmN+ClwO1lze3AdhGxV2Vd0sEUAeyOSs1HImKHyrqkQ4AngHsrNZ+IiGnl7brRmpWZ
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
KdYPPa38wN41LTV17TDGGkmSNMn03MLtFp8DXg28rtsd6aSIOBI4snps3rx5s4eHh7vUo7GbPr2f
bQcGut2NtvX39zMwBd7HZOKYN88xb55j3qy+vuIjZBctWnThihUrnmg5vSQzl4znuj0bkiJiMfAm
4PWZ+YvKqVUUm1QO8swZnEHgnkrNjIiY1TKbNFieG63ZsaXNacBAS82+LV0brJwb/XXwOWqeofxm
tX7D5gPL6up70YYN61mzZk23u9G2gYGBKfE+JhPHvHmOefMc82b19/czZ84choeHFwId+5SPnrzd
VgakvwTekJkPV89l5kMU4ePgSv0sinVEt5WHlgEbWmp2o1gAfnt56HZgu4jYq3L5gykC2B2Vmj0i
YodKzSHAE8C9lZoDy4BVrVmZma1pVpIkTRI9N5MUEZ+juBX1fwFPRsTorMwTmfnb8vcXAWdGxAPA
Tyg+IuUR4FooFnJHxGXABRGxlmJ/o4uBWzPzzrLm/ohYCnwxIk4AZlBsPbAkM0dngG6gCENXldsO
7FS2tTgz15c1VwMfB74cEecBewCnUDyBJ0mSJqlenEl6PzALuBl4tPJfjBZk5qcpAs2lFLM+WwEL
MvOpynUWAtcBX6tc6/CWto4C7qd4qu064BbgfZV2NlFsNLmRYpbqSuAKYLhSs45i5mhX4C7gfOCs
zLxsfG9fkiT1gr6RkZFu90GF+cCyVacczfoHV3a7L89uq63Z8twvsH6bWd3uSdtcN9A8x7x5jnnz
HPNmja5JothTcWqvSZIkSeo2Q5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5Ik
SVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVIN
Q5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5Ik
SVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVINQ5IkSVIN
Q5IkSVINQ5IkSVINQ5IkSVINQ5IkSVKN6d3uQJ2IeD1wGrA3sBNwWGb+Q0vN2cDxwHbArcAJmflA
5fxM4ALgHcBMYClwYmY+VqnZHlgMvAXYBFwDnJqZT1ZqdgE+DxwE/Aa4EjgjMzdVavYsr7Mv8Biw
ODPP78RYSJKk7ujVmaRtgH8BTgRGWk9GxOnAB4D3AvsBTwJLI2JGpewi4M3A4cCBwM4UIajqamB3
4OCy9kD8sfokAAANOUlEQVTg0ko7WwDfogiTBwDvBt4DnF2p2ZYigD0EzKcId2dFxPHjeeOSJKk3
9GRIysxvZ+bHM/NaoK+m5FTgnMy8LjN/BLyLIgQdBhARs4BjgYWZ+b3MvAcYAl4XEfuVNbsDhwLH
ZeZdmXkbcDJwRETMLds5FHgVcHRmLs/MpcDHgJMiYnQW7higv7zOfZmZwMXABzs7KpIkqUk9GZKe
TUS8DJgL3DR6LDPXAXcAry0P7UMx+1OtWQk8XKk5AFhbBqhRN1LMXO1fqVmemY9XapYCs4F5lZpb
MnNDS81uETF7nG9TkiR12aQLSRQBaQRY3XJ8dXkOYBB4qgxPm6uZS7F+6GmZuRFY01JT1w5jrJEk
SZPMZAxJkiRJE64nn257Dqso1ikN8swZnEHgnkrNjIiY1TKbNFieG63ZsXrhiJgGDLTU7NvS/mDl
3Oivg89R8wwRcSRwZPXYvHnzZg8PD9eV96Tp0/vZdmCg291oW39/PwNT4H1MJo558xzz5jnmzerr
K5YvL1q06MIVK1Y80XJ6SWYuGc91J11IysyHImIVxRNpP4SnF2rvD3y2LFsGbChrvl7W7Aa8FLi9
rLkd2C4i9qqsSzqYIoDdUan5SETsUFmXdAjwBHBvpeYTETGtvF03WrMyM1u/UaPvYQnQ+g2bX/Z7
UtiwYT1r1qzpdjfaNjAwMCXex2TimDfPMW+eY96s/v5+5syZw/Dw8ELg7k5dtydDUkRsA7yC3z/Z
9vKIeA2wJjN/RvF4/5kR8QDwE+Ac4BHgWigWckfEZcAFEbGWYn+ji4FbM/POsub+iFgKfDEiTgBm
AJdQJM7RGaAbKMLQVeW2AzuVbS3OzPVlzdXAx4EvR8R5wB7AKRRP4EmSpEmqV9ck7UNx62wZxSLt
z1Akw0UAmflpikBzKcWsz1bAgsx8qnKNhcB1wNeAm4FHKfZMqjoKuJ/iqbbrgFuA942eLDeMfAuw
EbiNYiPJK4DhSs06ipmjXYG7gPOBszLzsjbevyRJ6rK+kZE/2KtR3TEfWLbqlKNZ/+DKbvfl2W21
NVue+wXWbzOr2z1pm1PizXPMm+eYN88xb9bo7TaKT+ro2O22Xp1JkiRJ6ipDkiRJUg1DkiRJUg1D
kiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJ
Ug1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1D
kiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJ
Ug1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUg1DkiRJUo3p
3e7AVBERJwEfAuYCPwBOzsx/7m6vJEnSeDmT1AER8Q7gM8AwsBdFSFoaETt0tWOSJGncDEmdsRC4
NDOvzMz7gfcD/w4c291uSZKk8TIktSki+oG9gZtGj2XmCHAj8Npu9UuSJLXHkNS+HYBpwOqW46sp
1idJkqRJyIXbvWNLgK3efzoz/uPfu92XZ7fFFvRttQ3093e7J23r6+ujfwq8j8nEMW+eY948x7xZ
06c/HWe27Oh1O3mxF6jHgY3AYMvxQWBV3Qsi4kjgyOqxBQsWvHhoaIjZr95zQjqpzZszZ063u/CC
45g3zzFvnmPevMsvv/yS66+//ucth5dk5pLxXK9vZGSkA916YYuIfwLuyMxTy6/7gIeBizPz/Od5
mRddfvnlNwwNDZ0M/HaCuqoWixYtunB4eHhht/vxQuKYN88xb55j3rgtL7/88kuGhoYOAX7VqYs6
k9QZFwBXRMQy4E6Kp922Bq4YwzV+df311/98aGjotgnonzZjxYoVTwB3d7sfLySOefMc8+Y55s0r
f4Z2LCCBC7c7IjOTYiPJs4F7gD2BQzPzl13tmCRJGjdnkjokMz8HfK7b/ZAkSZ3hTJIkSVINQ1Jv
Gdfqe7XFMW+eY948x7x5jnnzOj7mPt0mSZJUw5kkSZKkGoYkSZKkGoYkSZKkGoYkSZKkGu6T1KCI
OIli08m5wA+AkzPzn5+l/iDgM8A8io85+e+Z+ZUGujpljGXMI+KtwAnAnwAzgRXAWZl5Q0PdnRLG
+ue88rrXATcDyzNz/oR2cooZx98tM4Bh4OjyNY8CZ2fmFRPf26lhHGN+NHAa8ErgCeB64LTMXNNA
dye9iHg9xfjtDewEHJaZ//AcrzmINn+GOpPUkIh4B8U3axjYi+J/qqURscNm6ncFrgNuAl4D/N/A
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Well that's not helpful. Looks like the amounts are so skewed we'll need to transform to visualize. Let's see if we can kill two birds with one stone using numpy's log1p.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[87]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">amount</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">amount</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">famount</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">dfraud</span><span class="o">.</span><span class="n">amount</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">amount</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">famount</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjAAAAFqCAYAAAAEDGBjAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAPYQAAD2EBqD+naQAAIABJREFUeJzt3X2cVWW9///X3swAgwg0gqDhTdgRlbSE6gtlZooZ9Sjv
Lw9aIXm8KQQj1OOpU3PA39G0g2be/DI10FT6Xl99HLOMNM3qVFSGZB00/WpqYUdEEQY4A8ye2d8/
1h7OzDAzzMzaw+w1vJ6PxzxgX+tan3XtWWzmPde6yxWLRSRJkrIk398DkCRJ6ikDjCRJyhwDjCRJ
yhwDjCRJyhwDjCRJyhwDjCRJyhwDjCRJyhwDjCRJyhwDjCRJyhwDjCRJypyq/h5AZ0IIc4BLgXHA
U8DcGOMTnfR9P3ANcBgwDHgZuDXG+PV2/c4EFgEHA88BV8QYl/dibDNjjMt6up4qk/tz4HGfDizu
z4GlXPuzImdgQghnAYuBOuBokgDzcAhhdCerbAFuBD5AEmKuBP6/EMI/tKr5PuBe4DbgXcD3gAdC
CEf0Yogze7GOKpf7c+Bxnw4s7s+BpSz7s1JnYOaTzKDcBRBCuAj4GPAZ4Nr2nWOMvwd+36rp3hDC
6SSB5vZS2zxgeYzxutLrr4QQTgQuBj7XJ+9CkiT1iYqbgQkhVANTgMda2mKMReBRYFo3axxd6vvT
Vs3TSjVae7i7NSVJUuWoxBmY0cAgYG279rXAxK5WDCH8FRhTWv9fYoxLWi0e10nNcalGK0mSdrtK
DDBpHAMMB6YC14QQno8x/u8yb2OfGTNmvBV4H7C1zLXVDyZNmjQSmNzf41D5uE8HFvfngDK09DN0
H+CNNIUqMcC8DjQBY9u1jwVe7WrFGOPLpb+uDiGMA/4FaAkwr/a0ZghhJu1ONpoxY8ZbZ8+ePRn4
ZVdjUXbU1dUBrOzvcah83KcDi/tzYJk9ezZLlix5ZPny5a+0W7SsJ1cnVVyAiTE2hhBWAicADwKE
EHKl19/oQalBwJBWr1d0UOPEUntnY1kGtP9mvg/45ZtvvkmhUOjBcFSpRowYQX19fX8PQ2XkPh1Y
3J8DR1VVFW95y1uYPXv23NmzZ/8qVa1yDarMrgOWloLMb0muShoGLAUIIVwN7B9jnFV6/TngL8Cf
Sut/EFgAtL4PzA3AT0MIXwAeIplZmQKc38OxbQUoFAo0Njb2+I2p8hSLRfflAOM+HVjcnwNS6lMw
Ku4qJIAYYyS5id0iYBVwFHBSjHFdqcs44IBWq+SBq0t9nwA+C1wWY6xrVXMFcDZwAckl16cBJ8cY
n+7bdyNJksotVywW+3sMWTMZWLlu3Tp/IxggamtrWb9+fX8PQ2XkPh1Y3J8DR3V1NWPGjIHkCMiT
aWpV5AyMJElSVwwwkiQpcwwwkiQpcyr1KiRJUj8bNWoU+Xz//56bz+epra3t72Gom5qbm9mwYUOf
b8cAI0nqUD6f9+RZ9djuCpv9H60lSZJ6yAAjSZIyxwAjSZIyxwAjSZIyxwAjSZIyxwAjSZK69NJL
LzF+/Hj+/d//vb+HsoMBRpK0R4kxMn78eA455BDWrl270/IzzjiD6dOn96r2nXfeSfI84u4ZP358
h1+TJ0/u1fb3JN4HRpLUY1Vb6mHzpv4dxPC9Kew1oterb9++nZtvvplFixaVbUh33XUXtbW1hBC6
vc4HP/hBzjjjjDZtQ4cOLduYBioDjCSp5zZvYttti/t1CEPOXwApAsykSZO45557uPjii9l3333L
OLKemTBhAqeeemqP1mloaKCmpqaPRpQNHkKSJO1xcrkcc+fOpampiZtuummX/Zuamrj++ut5//vf
z4QJE5g6dSpf/epX2b59+44+U6dO5dlnn2XFihU7DgWdeeaZqcc6d+5cjjjiCF588UU++clPMnHi
RD7/+c8DsGLFCi644ALe8573MGHCBN773veyaNEitm3b1qbGKaecwsyZMzus/f73v79N24YNG5g3
bx6HH344kyZNYsGCBWzevDn1+yg3Z2AkSXukAw88kDPOOIN77713l7MwCxYs4L777uPjH/84F154
IatWreKmm27ihRde4LbbbgNg0aJFfOlLX2L48OFccsklFItFRo8evctxbNu2badHNgwfPpzBgwfv
eN3Y2Mg555zD+973Purq6hg2bBgA3//+99m+fTuzZ89m1KhRPPnkk9xxxx289tprbYJZLpfrdPut
lxWLRc4991xWrVrFrFmzmDBhAj/84Q+ZP39+lzX6gwFGkrTHmjdvHvfddx8333wzCxcu7LDP008/
zX333cc555zDNddcA8CnP/1p9tlnH2699VZWrFjBtGnT+PCHP8w111xDbW0tp5xySrfHsGzZMu69
994dr3O5HNddd12b2ZutW7dy+umns2DBgjbr1tXVMWTIkB2vzz77bA444AAWL17Ml7/8ZcaOHdvt
cQD88Ic/5He/+x0LFy7kvPPO2/FeTzvttB7V2R08hCRJ2mMdeOCBnH766dxzzz2sW7euwz4/+clP
yOVynH/++W3aL7zwQorFIo899liqMZx00kl897vf3fG1bNkyjjvuuJ36fepTn9qprXV4aWhoYP36
9bz73e+mWCyyevXqHo/l8ccfZ8iQIZxzzjk72vL5PLNnz6ZYLPa4Xl9yBkaStEe75JJLuP/++7np
pps6nIVZs2YN+Xyet73tbW3ax4wZw8iRI1mzZk2q7e+3334cc8wxXfYZPHhwh4e41qxZw7XXXstj
jz3Gxo0bd7Tncjk2ber5VWJr1qxh3LhxO10Fdcghh/S4Vl8zwEiS9mgHHnggp512Gvfccw9z5szp
tF9/ngPS0WXVTU1NnHXWWWzZsoW5c+dyyCGHUFNTwyuvvMKCBQtobm7e0bezsbfukzUeQpIk7fEu
ueQSCoUCN998807Lxo8fT3NzM3/+85/btL/++uts3LiR8ePH72jbnSFn9erVvPzyyyxcuJCLLrqI
E088kWOOOabDmZqRI0dSX1+/U3v72aPx48fz6quvsnXr1jbtzz//fHkHXwYGGEnqZ1Vb6qla+0r5
v7bs/ANLHTvooIM47bTTuPvuu3c6F+b444+nWCxy++23t2m/9dZbyeVynHDCCTvaampqOgwKfSGf
T36Etz43pVgscscdd+wUpA466CCeffZZNmzYsKPtj3/8I08++WSbfscffzzbtm3j7rvv3tHW1NTE
kiVLvApJktROH90ULu2N3gayjk5InTdvHvfffz8vvPAChx122I72I444gjPPPJN77rmHjRs3MnXq
VFatWsV9993HjBkzmDZt2o6+Rx11FN/5zne44YYbOPjggxk9evRO91kpl4kTJ3LggQdSV1fHmjVr
2GuvvXjooYc6PPdl5syZ3HHHHZx99tmEEFi3bh333HMPEydObDPbMmPGDCZPnsyVV17Jyy+/zCGH
HMJDDz1EQ0NDn7yHNJyBkSTtcTqaTTj44IM5/fTTO1y2ePFiFixYwB/+8AcWLlzIihUrmDdvHrfc
ckubfvPnz+f444/nm9/8JhdffDFf//rXdzmO3s5sVFdXc+edd3L44Ydz4403csMNN3DooYdy3XXX
7dR34sSJ3HDDDWzcuJErr7ySn/zkJ9x0000cfvjhbbafy+W46667OPnkk7nvvvv42te+xoEHHthh
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
065lbZ9mMcBIndldAcboKkmSMscAI0mSMscAI0mSMscAI0mSMscAI0mSMscAI0mSMscAI0mSMscA
I0mSMscAI0mSMqdSn4UkSepnzc3NFfFAx3w+T3Nzc38PQ920u/aVAUaS1KENGzb09xAAH2mgjnkI
SZIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIk
ZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZU5Vfw9AktR3Nm1vpn57U1lqjRg8iL0H+3uvKoMBRpIG
sPrtTfz8zxvKUuvYCaMMMKoY/kuUJEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CR
JEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CRJEmZY4CRJEmZU7FPow4hzAEuBcYBTwFzY4xP
dNL3VOCzwLuAIcBq4F9ijI+063cmsAg4GHgOuCLGuLyv3oMkSeobFTkDE0I4C1gM1AFHkwSYh0MI
oztZ5VjgEWAGMBl4HPh+COGdrWq+D7gXuI0k6HwPeCCEcERfvQ9JktQ3KnUGZj5wa4zxLoAQwkXA
x4DPANe27xxjnN+u6UshhJOBj5OEH4B5wPIY43Wl118JIZwIXAx8rvxvQdJAVLWlHjZvKmvNXFOh
rPWkPUHFBZgQQjUwBbiqpS3GWAwhPApM62aNHLA3sL5V8zSSWZ3WHgZOTjVgSXuWzZvYdlv7/0rS
GTprTlnrSXuCigswwGhgELC2XftaYGI3a1wG7AXEVm3jOqk5rhdjlKQ9TnMxxyubG1PXGTF4EHsP
rsgzGJQhlRhgUgkhnA18GfhEjPH1/h6PJA0UWxqbWbVmY+o6x04YZYBRapUYYF4HmoCx7drHAq92
tWII4e+BbwFnxBgfb7f41Z7WDCHMBGa2bps0adLIuro6RowYQbFY7Go4yojq6mpqa2v7exgqo77c
pw1vrKWpalBZa+ZyearKXBOguqqKmqHDqKnZVr56NTWp69QMHUZt7d7d366f0QEjl8sBsHDhwutX
r17dPg0vizEu626tigswMcbGEMJK4ATgQdhxTssJwDc6W68UNm4Hzoox/qiDLis6qHFiqb2zsSwD
2n8zJwMr6+vraWxMP5Wq/ldbW8v69et33VGZ0Zf7tKpQoFBoKm/NYnPZawI0Fgo0bP1vGhoaylRv
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
58nnc50uz+Vy5PMtf3beDyDHrvvsXHvn/j2t01XNNLXa1y1XLWgZV54hNTWp6lRXVVHTgxr5/KBO
+/e0Vldqhg6jtnbvstRSx3K55N/iwoULr1+9evXGdouXxRiXdbdWxQWYGGNjCGElcALwIEAIIVd6
/Y1OVlsBzGjX9uFSe4dCCHngSOChLsayDGj/zZwMrKyvr6exsbGLd6KsqK2tZf369f09jD5RVShQ
KDSVt2axud9rNjc309zc+S8Q+Tw0NxfJF4td9gMosus+bWp3UrOndbqqmaZW+7rlqgUt42qmoaEh
VZ3GwuAe1aipqem0f09rdaVh6xDWr/f/9b5UXV3NmDFjqKurm09yLmqvVVyAKbkOWFoKMi2XUQ8D
lgKEEK4G9o8xzir1/yYwJ4RwDfBtkrBzBvDRloIhhC+THEJ6HhgFXA4cCNy+G96PJEkqo0o8iZcY
YyS5id0iYBVwFHBSq0ugxwEHtOr/Esll1tOB35MEnvNijK2vTHoL8C3gaZJZl+HAtBjjn/r0zUiS
pLKr1BkYYoy3kNyYrqNlszto+znJ5ded1fsC8IWyDVCSJPWbipyBkSRJ6ooBRpIkZY4BRpIkZY4B
RpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIk
ZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZY4BRpIkZU5Vfw9A0p5h85FT2TR4eIfLqvIjKEyZ3u1a
2/cdDy+8Wa6hScogA4yk3WLT4OH8rJPQUTX8TQo9CCRT9ptQrmFJyigPIUmSpMwxwEiSpMwxwEiS
pMwxwEiSpMzxJF6pAlRtqYfNm8peN9dUKHtNSaoEBhipEmzexLbbFpe97NBZc8peU5IqgYeQJElS
5hhgJElS5ngISZLUffkc+ZTna+W2D21bY/BgmgcPSTkw7WkMMJKk7mtspPDUE6lKFIdNofDkyh2v
qyZPAwOMeshDSJIkKXMMMJIkKXMMMJIkKXMMMJIkKXMMMJIkKXO8CklSpzYfOZVNg4e3aavKj6Aw
ZXqPa23fdzy88Ga5hiZpD2eAkdSpTYOH87N2oaNq+JsUehFEpuw3oVzDkiQPIUmSpOxxBkbqob54
crRPjZaknjHASD3VB0+O9qnRktQzHkKSJEmZU7EzMCGEOcClwDjgKWBujLHTB3CEEI4DFgOTgL8A
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Not only are we successsul in actually seeing the full scope of out data, we now know that log1p transformation gets us much closer to a normal distribution. Let's try the same thing with our other floats.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[89]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">oldbalanceOrg</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[89]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    5.090090e+05
mean     8.310125e+05
std      2.879680e+06
min      0.000000e+00
25%      0.000000e+00
50%      1.425700e+04
75%      1.075330e+05
max      5.039905e+07
Name: oldbalanceOrg, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[90]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dfraud</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">oldbalanceOrg</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[90]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    6.690000e+02
mean     1.598529e+06
std      3.478363e+06
min      0.000000e+00
25%      1.191438e+05
50%      4.475892e+05
75%      1.388952e+06
max      5.039905e+07
Name: oldbalanceOrg, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[91]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">oldbalanceOrg</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceOrg</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">foldbalanceOrg</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">dfraud</span><span class="o">.</span><span class="n">oldbalanceOrg</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">oldbalanceOrg</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">foldbalanceOrg</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
HZx77rlMmzat1b51dXX07t179+vx48dzwAEHMHv2bK6//noGDx6cdx4Ajz/+OC+++CLTp0/n4osv
3j3Wc845p6A43UEFTBc1/uKnNL69pqgxEyOOIX7uISpgRES6yYEHHsi55567ey3MoEGD2vT55S9/
SSwW45JLLmnVfumll3LnnXfy1FNPMWrUqC7ncMYZZ3DhhRe2ahs+fHibfl/5ylfatGUWLw0NDTQ0
NPDxj3+cVCrFqlWrCi5gnn76aXr37s0FF1ywuy0ejzNx4kRefPHFgmKVmgoYERHZo1155ZU8/PDD
zJ07t91ZmHXr1hGPxznkkENatQ8aNIj+/fuzbt26UMf/4Ac/yMknn9xhn169erV7imvdunXMmjWL
p556ii1btuxuj8Vi1NcXfpZg3bp1DBkypM1VUIceemjBsUpNBYyIiOzRDjzwQM455xwefPBBJk2a
lLNfT64Bae+y6mQyydixY9m+fTuTJ0/m0EMPpaamhnfeeYdp06bR3Ny8u2+u3DP7RI2uQhIRkT3e
lVdeSVNTE3fccUebbUOHDqW5uZm//OUvrdrfe+89tmzZwtChQ3e3dWeRs2rVKtauXcv06dO57LLL
GD16NCeffHK7MzX9+/dn69atbdqzZ4+GDh3K+vXr2bFjR6v21atXFzf5IlABIyIie7yDDjqIc845
hwceeICNGze22vbpT3+aVCrFPffc06r9rrvuIhaLcdppp+1uq6mpabdQKIV4PPgVnrluMpVKce+9
97YppA466CBee+01Nm/evLvtj3/8IytWrGjV79Of/jQ7d+7kgQce2N2WTCaZP3++rkISERHpae1d
LDFlyhQefvhh3njjDQ4//PDd7UceeSTnn38+Dz74IFu2bOGEE05g5cqVLFq0iDFjxrRawHvUUUfx
ox/9iFtvvZWDDz6YgQMHtrnPSrEMHz6cAw88kLq6OtatW8fee+/N4sWL2137Mm7cOO69917Gjx+P
mbFx40YefPBBhg8f3mq2ZcyYMYwcOZIbbriBtWvXcuihh7J48WIaGhpKMoYwNAMjIiJ7nPZmEw4+
+GDOPffcdrfNnj2badOm8Yc//IHp06ezbNkypkyZwrx581r1mzp1Kp/+9Ke58847ueKKK7jllls6
zaOrMxvV1dXcd999HHHEEdx+++3ceuutHHbYYcyZM6dN3+HDh3PrrbeyZcsWbrjhBn75y18yd+5c
jjjiiFbHj8Vi3H///Zx55pksWrSI73//+xx44IHtxuxpMV2yW7CRwPL1M68r0WXUF3b7oqra2to2
N1GKqkoaC2g85aySxgLtj6ejMVbCs5CkNDr6vqmurm65VP04YEW7nfKkU0giIlKwpr33CfUcIpGw
dApJREREIkcFjIiIiERO2Z5CMrNJwNXAEOBlYLK7v5Cj70nATcDhQB9gLXCXu9+S0WcCMB9IAS0r
lna4e5+SDUJERERKoixnYMxsLDAbqAOOJShglppZrsd6bgduB04hKGJuAP7dzL6W1W8LQUHU8nFQ
8bMXERGRUivXGZipBDMo9wOY2WXA54GLgFnZnd39JeCljKaHzOxcgoIm885DKXdvfYciERERiZyy
K2DMrJrg8qobW9rcPWVmTwJ5Pe7TzI5N9/121qa+ZraGYOZpBfCv7v5KMfIWERGR7lOOp5AGAglg
Q1b7BoLTPjmZ2dtmtgN4HrjD3ednbH6NYAbni8AFBGN/1sz2L1biIiIi0j3KbgYmpJOBvsAJwE1m
ttrdfwzg7s8Bz7V0NLNlwKvApQRrbURERCQiyrGAeQ9IAoOz2gcD6zva0d3Xpv+7ysyGAN8Ffpyj
b5OZrQQ+nCuemY0DxmW2jRgxon9dXR2JRIJUVaKjdAoWTyTo268fiURx43amurqa2trabj1mqVTS
WEDjKWeVNBZofzwtDwsUKUQ8Hs/5s9Hy2ILp06ffvGrVqi1Zmxe6+8J8j1N2BYy7N5rZcuA04FEA
M4ulX99WQKgE0DvXRjOLAx8DFneQy0Ig+5M5ElieTCZpakoWkE7nEskk9fX1epRACJU0FtB4ylkl
jQVyP0pApFDNzc2dPkqgrq5uKhX6KIE5wIJ0IfM8wVVJfYAFAGY2E9jf3SekX18OvAX8Kb3/J4Fp
QOZ9YK4nOIW0GhgAXAscSOurlERERMrGTTfdxLx581i7dm3nnfcwZTk/6O5OcBO7GcBK4CjgjIxL
oIcAB2TsEgdmpvu+AHwDuMbdM9e27Av8AHiFYNalLzDK3f+EiIjsUdydoUOHtvsxc+bMnk5vtzBP
q6505ToDg7vPA+bl2DYx6/VcYG4n8a4CripagiIiEmmxWIxrrrmGAw44oFX78OHDeygjKUTZFjAi
IlK+6nc1s3VXcdcBFmqfXgn69Qp3IuFTn/oUH/vYx/Lqm0ql2LVrF71751xeKd1IBYyIiBRs664k
v/7L5h7N4dRhA0IXMLkkk0kOOuggvva1r/HRj36UO+64gzVr1nDPPfdw2mmncccdd/DEE0+wevVq
duzYwfDhw5kyZQqf/exnd8dYs2YNJ598Mrfffjtnn312m9jXXnstU6ZM2d2+bNkyZsyYweuvv84H
P/hBJk2aVJKxVQoVMCIissfaunVrh1df/epXv+LRRx9lwoQJDBgwgA996EMA/PCHP+Rzn/sc55xz
Do2NjTzyyCNccsklPPDAA3zyk58sOI9Vq1bx5S9/mcGDB3PNNdewa9cuZs2axcCBuR4BKCpgRERk
j5RKpRg7dmyrtlgsxttvv7379ZtvvsnTTz/NIYcc0qrfs88+2+pU0oUXXsjo0aO5++67u1TAzJo1
i0QiwSOPPMJ+++0HwGc/+1lOP/103Y8nBxUwIiKyR4rFYtx4441tipNMJ598crvbM4uXLVu2kEwm
Of7441m6dGnBeTQ1NfHb3/6WL3zhC7uLF4DDDjuMU045hd/97ncFx9wTqIAREZE91jHHHNPhIt6h
Q4e22/7EE09w22238eqrr7Jz587d7b169So4h40bN7Jz504OPvjgNtsOPfRQFTA5qIARERHJYa+9
9mrT9rvf/Y6LL76Yk046iZkzZ7LffvtRVVXFQw89xOOPP767X677tySTPXv1VqVQASMiIlKAJUuW
0KdPHx588MFWz6574IEHWvXr378/EJxiyrRu3bpWrwcNGkTv3r1588032xxr9erVxUq74mhlkIiI
SAHi8TjxeLzVTMratWv5xS9+0arfgAED6N+/P7///e9btS9YsKDV7ExVVRWnnHIKS5YsYcOGDbvb
//SnP/Hb3/62RKOIPs3AiIjIHimVSnVpv9NPP50f/vCHjB8/nrPOOot3332X++67j0MPPZTXX3+9
Vd9x48Zx55130q9fPz72sY+xbNky1q5d2+bYV199NWeeeSZnnXUWX/3qV9m5cycLFizg8MMPbxNT
ApqBERGRPVJnzxjK9RyiU089le9///ts2LCBuro6HnvsMb7zne9w+umnt+k7bdo0vvSlL/HYY49x
4403kkgkuO+++9rE/uhHP8oDDzzAvvvuy3/+53+yaNEirrvuunZjSiDW1Qp0DzYSWL5+5nU0vr2m
qIETI44hfu6FNDc3FzVuZ2pra3M++jxqKmksoPGUs0oaC7Q/no7GWCmPEpDi6+j7prq6mkGDBgEc
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
TkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Looks like very few of our isFraud cases have a beginning balance from the source account of 0 when compared to all records.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[92]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">newbalanceOrig</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[92]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    5.090090e+05
mean     8.523704e+05
std      2.915303e+06
min      0.000000e+00
25%      0.000000e+00
50%      0.000000e+00
75%      1.448556e+05
max      4.367380e+07
Name: newbalanceOrig, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[93]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dfraud</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">newbalanceOrig</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[93]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    6.690000e+02
mean     1.637103e+05
std      1.938302e+06
min      0.000000e+00
25%      0.000000e+00
50%      0.000000e+00
75%      0.000000e+00
max      4.039905e+07
Name: newbalanceOrig, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[94]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">newbalanceOrig</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceOrig</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">fnewbalanceOrig</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">dfraud</span><span class="o">.</span><span class="n">newbalanceOrig</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">newbalanceOrig</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">fnewbalanceOrig</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAigAAAFqCAYAAAAwdaF/AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAPYQAAD2EBqD+naQAAIABJREFUeJzt3XucVXW9//HXHhhBUNARRA1vUJKSN7r8NLXyHpV57WOo
pWSWhWAc0uxUZw74O5qUmhc4mjdQCPumvzqakabZTTnlLfOQ6dG8RCZqCKhxnZnfH2tDwzADM7P3
zF6z5/V8PHjUfPda6/v5zox7v+e7vmutQlNTE5IkSXlSU+kCJEmSWjKgSJKk3DGgSJKk3DGgSJKk
3DGgSJKk3DGgSJKk3DGgSJKk3DGgSJKk3DGgSJKk3DGgSJKk3MllQImIQyLijoj4a0Q0RsTHN7P9
8RFxT0S8EhHLIuLBiDiqk32P61zV+eR48quaxgKOJ8+qaSzgePKsnGPJZUABBgK/B74ItOdhQR8A
7gHGAmOA+4E7I2LfTvRdNb8oRY4nv6ppLOB48qyaxgKOJ8/KNpa+5TpQOaWUfgr8FCAiCu3YfnKL
pq9FxLHAMcDj5a9QkiR1pbzOoJSkGGq2BpZUuhZJktRxVRlQgPPIThOlShciSZI6LpeneEoREacA
3wA+nlJ6rYO7bzd27Ni3Ae8HVpa9uAoYPXr0YLJ1OVWhmsZTTWMBx5Nn1TQWcDw51r/4Gbod8PdS
D1ZoamrPGtTKiYhG4LiU0h3t2PaTwPXAScV1LJvadhwtFvOMHTv2bePHj6+GXxJJkiripptuenT+
/Pl/bdE8L6U0ryPHqZqAUgwc1wMnp5R+3Mnu3g888Prrr7N27dpOHiJfBg0axPLlyytdRtlU03iq
aSzgePKsmsYCjiev+vbty7bbbgtwEPBgyccruaIuEBEDgbcD667gGVG8ZHhJSukvEXExsFNK6fTi
9qcAs4BJwEMRMay434qUUkd+6isB1q5dy5o1a8owkspramqqmrFAdY2nmsYCjifPqmks4Hh6gLIs
kcjrItn3AI8Bj5DdB+VS4FFgavH1HYCdm21/FtAHmAG81Ozfd7qpXkmSVEa5nEFJKf2STYSnlNL4
Fl8f2uVFSZKkbpPXGRRJktSLGVAkSVLuGFAkSVLu5HINiiSpa22zzTbU1PTMv1Framqoq6urdBll
05PG09jYyNKlS7ulLwOKJPVCNTU1LFni48rUMd0ZpHpmfJYkSVXNgCJJknLHgCJJknLHgCJJknLH
gCJJknLHgCJJUi/2/PPPM3z4cH74wx9WupQNGFAkSVUjpcTw4cMZOXIkixcv3uj1k046iSOOOKJT
x549ezYppXZvP3z48Fb/jRkzplP99zbeB0WStIG+by2HN9+obBFbbc3agYM6vfvq1auZMWMG06ZN
K1tJN998M3V1dUREu/f54Ac/yEknnbRBW//+/ctWUzUzoEiSNvTmG6y67tKKltDvrClQQkAZPXo0
c+fO5ZxzzmH77bcvY2UdM2LECI4//vgO7bNixQq23HLLLqqo5/AUTyueW7qKP722sqz/XnpzbaWH
JUm9QqFQYOLEiTQ0NHD11VdvdvuGhgYuv/xyDjroIEaMGMEBBxzAN7/5TVavXr1+mwMOOICnnnqK
BQsWrD9V84lPfKLkWidOnMhee+3Fc889x2mnncaoUaP40pe+BMCCBQv43Oc+x3vf+15GjBjB+973
PqZNm8aqVas2OMZxxx3HuHHjWj32QQcdtEHb0qVLmTRpEnvuuSejR49mypQpvPnmmyWPoys4g9KK
p179B68sX1HWY44aOpCdtvLbLUndYZddduGkk07ie9/73mZnUaZMmcJtt93GMcccw+c//3kee+wx
rr76ap599lmuu+46AKZNm8bXvvY1ttpqK84991yampoYMmTIZutYtWrVRo8U2Gqrrdhiiy3Wf71m
zRpOPfVU3v/+91NfX8+AAQMAuPPOO1m9ejXjx49nm2224dFHH+WGG27glVde2SB4FQqFNvtv/lpT
UxNnnHEGjz32GKeffjojRozgJz/5CZMnT97kMSrFT0xJUlWaNGkSt912GzNmzGDq1KmtbvPHP/6R
2267jVNPPZVLLrkEgE9/+tNst912XHvttSxYsIADDzyQo446iksuuYS6ujqOO+64dtcwb948vve9
763/ulAocNlll20w+7Jy5UpOPPFEpkyZssG+9fX19OvXb/3Xp5xyCjvvvDOXXnop3/jGNxg2bFi7
6wD4yU9+wsMPP8zUqVM588wz14/1hBNO6NBxuouneCRJVWmXXXbhxBNPZO7cubz66qutbvPzn/+c
QqHAWWedtUH75z//eZqamrjvvvtKquHoo4/m1ltvXf9v3rx5fOhDH9pou0996lMbtTUPJytWrGDJ
kiW85z3voampiYULF3a4lvvvv59+/fpx6qmnrm+rqalh/PjxNDU1dfh4Xc0ZFElS1Tr33HO5/fbb
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
tcH7fN++fTnkkEOYP38+//qv/8qwYdlH1Z/+9Cd+85vf9LifQ01NTZv/Xawb99SpUy9fuHDhshYv
z0spzetIX9USUBYAY1u0HVVsb1XxG9XymzUGeGTVqlWsWLGirAWuXFnD66+/XtZjtkddXd1Gl9n1
ZNU0nmoaCziePGttLNUSvtrS1NTUqf2OOOIIbrzxRk455RSOO+44XnnlFWbPns3IkSN5+umnN9h2
3LhxXHPNNWy99dbsvffeLFiwgBdeeGGjvr/85S9z7LHHctxxx/HpT3+aVatWMWvWLN75zndudMy8
a2xsbPO/i9raWoYOHUp9ff1k4NFS+8plQImIgcDbgXUxdERE7AssSSn9JSIuBnZKKZ1efP0aYEJE
XALcCBwOnAR8pJtLlyTlwOZmq9t6Ds4HPvABvvWtbzFz5kzq6+vZdddd+bd/+zeeffbZjcLElClT
WLp0KT/+8Y+58847OeKII5g9ezb777//Bsd+17vexZw5c7jwwgv59re/zY477sgFF1zAiy++2OMC
SncqdDZldqWI+CBwP9CyuNkppc9ExE3Arimlw5rt8wGyq3b2AhYB01JKt3Sw6zHAI7c8+DSvLC/v
DMqooQMZs+OAsh6zParpr0CorvFU01jA8eRZWzMobY2vWm51r/Lb1O/NuhkUspusVucMSkrpl2zi
CqOU0vhW2n5F9k2RJJVg6y1qDAeqOH8DJUlS7hhQJElS7hhQJElS7hhQJElS7hhQJElS7hhQJElS
7hhQJElS7hhQJElS7hhQJElS7uTyTrKSpK7V2NjYYx8YWFNTQ2NjY6XLKJueNJ7urNOAIkm90NKl
SytdQqdV03OSoPrGUy6e4pEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSblj
QJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEk
SbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSbljQJEkSblj
QJEkSbljQJEkSbljQJEkSbnTt9IFtCUiJgBfBnYAHgcmppQe2sT2pwLnAe8AlgHzgfNSSku6oVxJ
klRGuZxBiYiTgUuBemB/soByd0QMaWP7g4DZwHXAXsBJwPuA73ZLwZIkqazyOoMyGbg2pXQzQESc
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
IkmScseAIkmScseAIkmScseAIkmScseAIkmScseAIkmScseAIkmScseAIkmScseAIkmScseAIkmS
cseAIklXlwI4AAAOmElEQVSScseAIkmScseAIkmScseAIkmScseAIkmScqdvpQtoS0RMAL4M7AA8
DkxMKT3UxrbHA18A9gP6AQuBf08p3dNN5UqSpDLK5QxKRJwMXArUA/uTBZS7I2JIG7t8ALgHGAuM
Ae4H7oyIfbuhXEmSVGZ5nUGZDFybUroZICLOBj4KfAaY3nLjlNLkFk1fi4hjgWPIwo0kSepBcjeD
EhG1wLuB+9a1pZSagHuBA9t5jAKwNbCkK2qUJEldK3cBBRgC9AEWt2hfTLYepT3OAwYCqYx1SZKk
bpLXUzydFhGnAN8APp5Seq3S9UiSpI7LY0B5DWgAhrVoHwa8vKkdI+KTwHeBk1JK929m23HAuOZt
o0ePHlxfX09NoUBNTaHDhW9KTU2BQYMG0adPn7Ied3Nqa2upq6vr1j67UjWNp5rGAo4nz6ppLOB4
8qpQyD43p06devnChQuXtXh5XkppXkeOl7uAklJaExGPAIcDd8D6NSWHA1e2tV8xcFwPnJxS+mk7
+pkHtPxmjQEeaWxqorGxqZMjaF1jYxPLly+nqam8x92curo6liypnqU41TSeahoLOJ48q6axgOPJ
q9raWoYOHUp9ff1k4NFSj5e7gFJ0GTCrGFR+R3ZVzwBgFkBEXAzslFI6vfj1KcXXJgEPRcS62ZcV
KaXl3Vu6JEkqVR4XyZJSSmQ3aZsGPAbsAxydUnq1uMkOwM7NdjmLbGHtDOClZv++0101S5Kk8snr
DAoppZnAzDZeG9/i60O7pShJktQtcjmDIkmSejcDiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJ
yh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0D
iiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyh0DiiRJyp2+lS5A
ktQ+fd9aDm++0eH9Vvx9MX3Xrm17g622Zu3AQSVUJpWfAUWSeoo332DVdZd2eLeGvn1Yu7ahzdf7
nTUFDCjKGQOKJPVyhT596Lv4r11zcGdn1EkGFEnq7Va8xarZM7rk0M7OqLNcJCtJknLHgCJJknLH
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
RU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Wow! For the isFraud cases, it looks like the updated balance from the original account is (nearly) always 0!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[95]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">oldbalanceDest</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[95]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    5.090090e+05
mean     1.097301e+06
std      3.392718e+06
min      0.000000e+00
25%      0.000000e+00
50%      1.294440e+05
75%      9.375382e+05
max      2.511150e+08
Name: oldbalanceDest, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[96]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dfraud</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">oldbalanceDest</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[96]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    6.690000e+02
mean     6.201907e+05
std      2.015292e+06
min      0.000000e+00
25%      0.000000e+00
50%      0.000000e+00
75%      1.883258e+05
max      2.429635e+07
Name: oldbalanceDest, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[100]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">oldbalanceDest</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span>
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">foldbalanceDest</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">dfraud</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">oldbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span>
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">oldbalanceDest</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">foldbalanceDest</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
nqF3795cdtll29uy2Szjxo0jl8u1ebxScwZFklS2rr/+eh555BHuvvvuJmdRli9fTjab5cADD9yh
vaqqioEDB7J8+fKijr/33ntz0kkntdinV69eTV6CWr58OdOmTWPRokWsXbt2e3smk2H9+rY/ZbV8
+XKGDh2601NEw4cPb/NYHcGAIkkqW/vttx8XXngh8+bNY/z48c3268x7MJp67Liuro5LLrmEDRs2
MGHCBIYPH07fvn159913mTx5MvX19dv7Nld7wz5dkZd4JEll7frrr2fr1q3MmDFjp23Dhg2jvr6e
P/7xjzu0v//++6xdu5Zhw4Ztb+vIELNs2TLeeecdampquO666zjrrLM46aSTmpxpGThwIOvWrdup
vfHsz7Bhw3jvvffYtGnTDu1vvPFGssUnxIAiSSpr+++/PxdeeCEPPfTQTveinH766eRyOe67774d
2u+55x4ymQxnnHHG9ra+ffs2GQRKIZvN/3hueG9ILpfj/vvv3yko7b///rz22musWbNme9tvf/tb
li5dukO/008/ndraWh566KHtbXV1dcyaNcuneCRJKrWmbvicOHEijzzyCG+++SaHHnro9vbDDz+c
L3zhC8ybN4+1a9dy/PHH8/LLL7NgwQJGjx7NCSecsL3vUUcdxYMPPsgdd9zBAQccwODBg3daZyQp
I0aMYL/99qO6uprly5ez22678fjjjzd578mYMWO4//77ufTSSwkhsGrVKubNm8eIESN2mC0ZPXo0
o0aN4qabbuKdd95h+PDhPP7442zcmOzCpElxBkWSVFaamg044IADuOiii5rcNn36dCZPnsxvfvMb
ampqWLx4MRMnTmTmzJk79Js0aRKnn3463//+9/nqV7/K9773vV3W0d6ZiYqKCubMmcNhhx3GXXfd
xR133MEhhxzCbbfdtlPfESNGcMcdd7B27Vpuuukmfvazn3H33Xdz2GGH7XD8TCbD3Llz+fznP8+C
BQu49dZb2W+//ZocMw0yaXy0qBONApY8+PzrJVnq/mCXuu9wlZWVOy2SpK7Nc5qMlr6P5fAuHpVG
S39vKioqqKqqAjgWWNpkpzbwEo8kaQdbdxtQ1HtwpCR4iUeSJKWOAUWSJKWOAUWSJKWOAUWSJKWO
AUWSJKWOAUWSJKWOAUWSJKVOatdBCSGMB24AhgKvAhNijC+20L8XUA1cVtjnL8CUGOPs0lcrSZKS
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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[102]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">newbalanceDest</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[102]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    5.090090e+05
mean     1.219353e+06
std      3.651259e+06
min      0.000000e+00
25%      0.000000e+00
50%      2.121075e+05
75%      1.106533e+06
max      2.506381e+08
Name: newbalanceDest, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[103]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dfraud</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">include</span><span class="o">=</span> <span class="kc">None</span><span class="p">)</span><span class="o">.</span><span class="n">newbalanceDest</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[103]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>count    6.690000e+02
mean     1.403334e+06
std      2.918224e+06
min      0.000000e+00
25%      0.000000e+00
50%      1.087290e+04
75%      1.183814e+06
max      2.529446e+07
Name: newbalanceDest, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[101]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">newbalanceDest</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Not Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span>
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">fnewbalanceDest</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">dfraud</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">label</span> <span class="o">=</span> <span class="s1">&#39;Fraud&#39;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> 
                            <span class="nb">range</span> <span class="o">=</span> <span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">min</span><span class="p">(),</span> <span class="n">np</span><span class="o">.</span><span class="n">log1p</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">newbalanceDest</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()),</span>
                            <span class="n">bins</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="n">normed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span> 

<span class="n">newbalanceDest</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
<span class="n">fnewbalanceDest</span><span class="o">.</span><span class="n">legend</span><span class="p">();</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAigAAAFqCAYAAAAwdaF/AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAAPYQAAD2EBqD+naQAAIABJREFUeJzt3Xuc1XW97/HXWjDcFRxBUPEGJSplitnR1LygKXUsb33d
qKWjme4IjINWe7fbE3i2Jm00Fd251QAF8Xy3ntqasTXNamuUCmZFqccbhiViyDVuM7POH2tBM8MM
zMzvN7N+rHk9Hw8eOd/f9/f9fmZ+LebN93fLFQoFJEmSsiRf7gIkSZKaM6BIkqTMMaBIkqTMMaBI
kqTMMaBIkqTMMaBIkqTMMaBIkqTMMaBIkqTMMaBIkqTMMaBIkqTMyWxACSFMCCG8HkLYEEL4ZQjh
6J307xVC+JcQwhshhI0hhNdCCJd0YN7xHS5amePxrDwe08ri8awsaR7PTAaUEML5wAygFjgSeAF4
NIQweAe7/QdwMlADHAyMB17qwPR+WCqLx7PyeEwri8ezsqR2PHumNVDKJgN3xBjvAQghXAl8ErgU
mN68cwjhDOAEYESMcVWp+c0uqlWSJKUscwElhFAFHAVct7UtxlgIITwOHNvKbmcCzwFfDSF8FlgP
PAR8I8a4sZNLliRJKctcQAEGAz2A5c3alwOjWtlnBMUVlI3AWaUx/g2oBi7rnDIlSVJnyWJA6Yg8
0ABcEGNcBxBC+F/Af4QQvhhj3NTGcfYcN27cvsBHKYYd7eJGjx49EBhT7jqUHo9pZfF4VpQ+pd+h
ewJ/STpYFgPKu0A9MLRZ+1Dg7Vb2+TPw1tZwUvIHIAcMB15tvkPpSuMmF/OMGzdu35qamjHA0x0r
XVlTW1sLsKjcdSg9HtPK4vGsLDU1NcyaNeuxBQsWvNVs0/wY4/z2jJW5gBJj3BJCWASMpXgdCSGE
XOnrW1rZ7WngvBBCvxjjX0ttoyiuqixrZZ75QPMf1keBp9977z3q6uqSfSPKhN133501a9aUuwyl
yGNaWTyelaNnz57sscce1NTUTKypqflF4vHSKKoT3AjMLgWVZyje1dMPmA0QQrge2CfGeHGp/33A
PwGzQgjfBIZQvNvn7nac3oHSaZ26ujq2bNmSwrehcisUCh7LCuMxrSwez4qUyiUSmXwOSowxAlcD
04DngcOB02OMK0pdhgH7Neq/HjgNGAQ8C9wL/CdwVReWLUmSUpIrFArlriFLxgCLVqxYYaKvENXV
1axcubLcZShFHtPK4vGsHFVVVQwZMgSKjwpZnHS8TK6gSJKk7s2AIkmSMseAIkmSMierd/FIkjrR
oEGDyOfL/2/UfD5PdXV1uctQGzU0NLBq1aqdd0yBAUWSuqF8Pu/FqWq3rgyT5Y/PkiRJzRhQJElS
5hhQJElS5hhQJElS5hhQJElS5hhQJEnqxt544w2GDx/O97///XKX0oQBRZJUMWKMDB8+nJEjR7J8
+fLttp933nmceuqpHRp7zpw5FN9l2zbDhw9v8c+YMWM6NH9343NQJElN9Fy/BtatLW8RA3ajrv/u
Hd598+bN3HbbbUybNi21ku655x6qq6sJIbR5nxNPPJHzzjuvSVufPn1Sq6mSGVAkSU2tW8umO2eU
tYTel0+BBAFl9OjRzJs3jy996UvstddeKVbWPiNGjODss89u1z4bNmygb9++nVTRrsNTPJKkipLL
5Zg4cSL19fXMnDlzp/3r6+u56aabOO644xgxYgTHHHMM3/rWt9i8efO2PscccwwvvfQSCxcu3Haq
5jOf+UziWidOnMhhhx3G66+/zkUXXcSoUaP48pe/DMDChQv5whe+wNFHH82IESP4yEc+wrRp09i0
aVOTMc466yzGjx/f4tjHHXdck7ZVq1YxadIkDj30UEaPHs2UKVNYt25d4u+jM7iCIkmqOPvvvz/n
nXce9913305XUaZMmcIDDzzAmWeeyRVXXMHzzz/PzJkzefXVV7nzzjsBmDZtGl//+tcZMGAAV111
FYVCgcGDB++0jk2bNm33SoEBAwbQq1evbV9v2bKFCy+8kI9+9KPU1tbSr18/AB5++GE2b95MTU0N
gwYNYvHixdx999288847TYJXLpdrdf7G2wqFApdccgnPP/88F198MSNGjOBHP/oRkydP3uEY5WJA
kSRVpEmTJvHAAw9w2223MXXq1Bb7/P73v+eBBx7gwgsv5IYbbgDgc5/7HHvuuSd33HEHCxcu5Nhj
j+XjH/84N9xwA9XV1Zx11lltrmH+/Pncd999277O5XLceOONTVZfNm7cyLnnnsuUKVOa7FtbW0vv
3r23fX3BBRew3377MWPGDL7xjW8wdOjQNtcB8KMf/YjnnnuOqVOnctlll237Xs8555x2jdNVPMUj
SapI+++/P+eeey7z5s1jxYoVLfb5yU9+Qi6X4/LLL2/SfsUVV1AoFHjiiScS1XD66adz//33b/sz
f/58TjrppO36ffazn92urXE42bBhAytXruTDH/4whUKBJUuWtLuWJ598kt69e3PhhRdua8vn89TU
1FAoFNo9XmdzBUWSVLGuuuoqHnzwQWbOnNniKsqyZcvI5/McdNBBTdqHDBnCwIEDWbZsWaL59957
b44//vgd9unVq1eLp6CWLVvG9OnTeeKJJ1i9evW29lwux9q17b/LatmyZQwbNmy7u4hGjhzZ7rG6
ggFFklSx9t9/f8455xzmzZvHhAkTWu1XzmswWrrtuL6+nvPPP5/169czceJERo4cSd++fXnrrbeY
MmUKDQ0N2/q2VnvjPrsiT/FIkiraVVddRV1dHbfddtt224YPH05DQwOvvfZak/Z3332X1atXM3z4
8G1tXRlilixZwtKlS5k6dSpXXnklp512Gscff3yLKy0DBw5kzZo127U3X/0ZPnw4b7/9Nhs3bmzS
/sorr6RbfEoMKJKkinbAAQdwzjnnMHfu3O2uRTnllFMoFArcddddTdrvuOMOcrkcY8eO3dbWt2/f
FoNAZ8jni7+eG18bUigUuPvuu7cLSgcccAAvvfQSq1at2tb229/+lsWLFzfpd8opp7Bp0ybmzp27
ra2+vp5Zs2Z5F48kSZ2tpQs+J02axIMPPsirr77KIYccsq39sMMO4zOf+Qzz5s1j9erVHHPMMTz/
/PM88MADjBs3jmOPPXZb38MPP5x7772Xm2++mQMPPJDBgwdv95yRtIwaNYr999+f2tpali1bRv/+
/XnkkUdavPZk/Pjx3H333VxwwQWEEFixYgXz5s1j1KhRTVZLxo0bx5gxY7j22mtZunQpI0eO5JFH
HmHDhg2d8j0k5QqKJKmitLQacOCBB3Luuee2uG3GjBlMmTKF3/zmN0ydOpWFCxcyadIkbr/99ib9
Jk+ezCmnnMJ3v/tdvvSlL/Gd73xnp3V0dGWiqqqKOXPmcOihh3Lrrbdy8803c/DBB3PjjTdu13fU
qFHcfPPNrF69mmuvvZaf/OQnzJw5k0MPPbTJ/LlcjnvuuYdPf/rTPPDAA3z7299m//33b3HMLMhl
8daiMhoDLFqxYgVbtmwpdy1KQXV19XYPSdKuzWOajh39HCvhXTzqHDv6/01VVRVDhgwBOApY3GKn
dvAUjySpibr+uyd6D46UBk/xSJKkzDGgSJKkzDGgSJKkzDGgSJKkzDGgSJKkzDGgSJKkzDGgSJKk
zDGgSJKkzDGgSJKkzDGgSJKkzDGgSJJUBjfccAMHHHBAucvILAOKJKmixBgZPnx4i3+uv/76cpe3
TZK3HXcHvixQklRxcrkc11xzDfvtt1+T9lGjRpWpIrWXAUWS1MTazQ2s2Vxf1hp279WD3XolW+Q/
+eST+eAHP9imvoVCgc2bN9O7d+9Ecyo9BhRJUhNrNtfz89dWlbWGj40YlDigtKa+vp4DDjiAz3/+
83zgAx/gtttu44033uCuu+5i7Nix3HbbbTz22GO88sorbNy4kVGjRjFp0iTOOOOMbWO88cYbHH/8
8dx6662cffbZ2439la98hUmTJm1rX7hwIdOmTePll19m7733ZsKECZ3yvVWSzAaUEMIE4GpgGPAC
MDHG+GwrfU8EnmzWXAD2jjG+05H50z4vWCgUUh1PkrRja9asYeXKlU3aqqurt/33z372Mx566CEu
vvhiBg0axL777gvA9773PT7xiU9wzjnnsGXLFn7wgx9w+eWXM3fuXE488cR217FkyRIuuugihg4d
yjXXXMPmzZuZPn06gwcPTvYNVrhMBpQQwvnADOALwDPAZODREMLBMcZ3W9mtABwMrN3a0NFw8sLb
f2XVXzd1ZNdWHTCoD/vulskftyRVnEKhwPnnn9+kLZfL8cc//nHb16+//jpPPvkkBx10UJN+v/jF
L5qc6rnkkks47bTTuPPOOzsUUKZPn06PHj34wQ9+wF577QXAGWecwamnnko+770qrcnqb8zJwB0x
xnsAQghXAp8ELgWm72C/FTHGNUkn//PaTbyzZkPSYZrYq38V2f1xS1JlyeVyXHfddduFj8aOP/74
Frc3DierV6+mvr6eo48+mkcffbTdddTV1fHUU09x5plnbgsnAAcffDAnnHACTz/9dLvH7C4y9xsz
hFAFHAVct7UtxlgIITwOHLuDXXPAr0MIfYDfAd+MMf6iU4uVJGXWEUccscOLZIcPH95i+2OPPcYt
t9zCH/7wBzZt+ttqeq9evdpdw4oVK9i0aRMHHnjgdttGjhxpQNmBLK4tDQZ6AMubtS+neD1KS/4M
XAGcC5wD/BH4aQjhiM4qUpK0a+vTp892bU8//TSXXXYZAwYM4Prrr2fu3Lncf//9fOpTn6KhoWFb
v9auU6yvL+/dT5UkcysoHRFjfBl4uVHTL0MIIymeKrq4PFVJknY1CxYsoF+/fsybN48ePXpsa587
d26TfgMHDgSKp4AaW7ZsWZOvhwwZQu/evXn99de3m+uVV15Jq+yKlMWA8i5QDwxt1j4UeLsd4zwD
HNfaxhDCeGB847bRo0cPrK2tpXfv3vTt246Z2qBPnz7ssccgnxrYxaqqqppcta9dn8c0HV6c2bJ8
Pk8+n6e+vn5bQFm6dCk//vGPm/QbNGgQAwcO5Fe/+hWXXHLJtvbZs2c3+Xu+Z8+enHDCCSxYsIB/
/Md/ZOjQ4q+2F198kaeeemqXOw75fL7Vz9/W73vq1Kk3LVmyZHWzzfNjjPPbM1fmAkqMcUsIYREw
FngIIISQK319SzuGOoLiqZ/W5pkPNP9hjQEWbdq0iQ0b0r1IduPGKt57771Ux9TOVVdXb3eboXZt
HtN0VHrI6+ijHU499VS+973vccEFF3DWWWfxzjvvMGfOHEaOHMnLL7/cpO/48eP57ne/y2677cYH
P/hBFi5cyNKlS7eb++qrr+bTn/40Z511Fp/73OfYtGkTs2fP5pBDDtluzKxraGho9fNXVVXFkCFD
qK2tnQwsTjpX5gJKyY3A7FJQ2XqbcT9gNkAI4XpgnxjjxaWvrwJeB5YAfYDLgZOB07q8cklS2e1s
tbq19+B87GMf49vf/ja33347tbW1HHDAAfzzP/8zr7766nZhYsqUKaxatYof/vCHPPzww5x66qnM
mTOHI488ssnYH/jAB5g7dy7XXnst//qv/8ree+/N1772Nd58881dLqB0pVxWHyAWQvgi8BWKp3Z+
TfFBbc+Vts0CDogxnlL6+hqKz0zZB/gr8Btgaozx5+2cdgyw6N5fvJz6bcZHD9+d91X7COWu5r+2
K4/HNB07+jlWyqPulb4d/f9m6woKxTtxE6+gZDaglIkBpcL4y6zyeEzT4c9RHdGVAcV4KkmSMseA
IkmSMseAIkmSMseAIkmSMseAIkmSMseAIkmSMseAIkmSMseAIkmSMseAIkmSMier7+KRJHWihoaG
TLwwMJ/P09DQUO4y1EZdeawMKJLUDa1atarcJQA+cl+t8xSPJEnKHAOKJEnKHAOKJEnKHAOKJEnK
HAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOK
JEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnKHAOKJEnK
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
qtsxoEhSd7RuLZvunJH6sL0vnwIGFKXAgCJJGdVpqxxArr6uU8aV0mJAkaSs6qRVDoA+F0/olHGl
tHiRrCTtfqLpAAAMNElEQVRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwD
iiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJyhwDiiRJ
ypye5S5AktT1VvfajZVHnZr6uHv07Ef/1EdVd2RAkaRuaE1djp+9+l7q455yZMGAolR4ikeSJGVO
ZldQQggTgKuBYcALwMQY47Ot9B0GzAA+DLwPuDnG+L+6qlZJkpSuTAaUEML5FAPHF4BngMnAoyGE
g2OM77awS2/gHeDaUl9JUhnk8nl6Ln+rzf03/GU5Pevq2tZ5wG7U9d+9g5VpV5PJgEIxZNwRY7wH
IIRwJfBJ4FJgevPOMcalpX0IIVzWhXVKkhrbsoVNs2a0uXt9zx7U1dW3qW/vy6eAAaXbyNw1KCGE
KuAo4ImtbTHGAvA4cGy56pIkSV0niysog4EewPJm7cuBUV1fjiSpreqrevPndty+nM/naWhoaFNf
b2HuXrIYUCRJu6j1dfBsO25fzudzNDQU2tTXW5i7lywGlHeBemBos/ahwNtpTRJCGA+Mb9w2evTo
gbW1teRzOfL5XFpTAcV/JQwaNIh8PnNn1SpaVVUV1dXV5S5DKepOx3TDX5ZT37NHp4yd64S/54oD
065xi3W0rW+PfL7bHPtdUS5XPO5Tp069acmSJaubbZ4fY5zfnvEyF1BijFtCCIuAscBDACGEXOnr
W1KcZz7Q/Ic1BljUUCi0OdG3VUNDA6tWrUp1TO1cdXU1K1euLHcZSlF3OqY96+rafAFpexU64e+5
4sC0a9x8vu396xsaus2x3xVVVVUxZMgQamtrJwOLk46XuYBSciMwuxRUtt5m3A+YDRBCuB7YJ8Z4
8dYdQggfAnLAAGBI6evNMcY/dHHtkrqZnuvXwLq1qY+bq2/j7bdSBcpkQIkxxhDCYGAaxVM7vwZO
jzGuKHUZBuzXbLfnga0xfAxwAbAUGNH5FUvq1tatZdOdbb+1tq36XDwh9TGlXUUmAwpAjPF24PZW
ttW00ObFHZIkVYjMBhRJkhqr79mLt9ZtSX3c3Xv1YLde/hs3awwokqRdwvq6AotfS/9mg4+NGGRA
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
/wEN2hAy9kSG5QAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="What-do-we-have-left-to-analyze?-Or,-I-am-sick-of-scrolling.">What do we have left to analyze? Or, I am sick of scrolling.<a class="anchor-link" href="#What-do-we-have-left-to-analyze?-Or,-I-am-sick-of-scrolling.">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[105]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 509009 entries, 0 to 509008
Data columns (total 11 columns):
step              509009 non-null int64
type              509009 non-null object
amount            509009 non-null float64
nameOrig          509009 non-null object
oldbalanceOrg     509009 non-null float64
newbalanceOrig    509009 non-null float64
nameDest          509009 non-null object
oldbalanceDest    509009 non-null float64
newbalanceDest    509009 non-null float64
isFraud           509009 non-null int64
id                509009 non-null int64
dtypes: float64(5), int64(3), object(3)
memory usage: 42.7+ MB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>So far, we've worked through step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest and newbalanceDest. All that's left are nameOrig and nameDest. Given that these are basically unique account numbers, there may not be much to glean. But wait! We can distinguish what type of account (merchant or consumer) by the leading character.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[106]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Let&#39;s create a lambda function and a new variable to analyze the account types</span>
<span class="n">df</span><span class="p">[</span><span class="s1">&#39;nameOrigCat&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">nameOrig</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[108]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">nameOrigCat</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[108]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>C    509009
Name: nameOrigCat, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Hmm, looks like all transactions started from consumer accounts regardless of isFraud status. Let's try namedest:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[109]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;nameDestCat&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">nameDest</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[111]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">nameDestCat</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[111]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>C    336292
M    172717
Name: nameDestCat, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Huzzah! Looks like there's some distinction there.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[114]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pd</span><span class="o">.</span><span class="n">crosstab</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">isFraud</span><span class="p">,</span> <span class="n">df</span><span class="o">.</span><span class="n">nameDestCat</span><span class="p">)</span> <span class="c1"># plot the crosstab for survival by sex</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[114]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nameDestCat</th>
      <th>C</th>
      <th>M</th>
    </tr>
    <tr>
      <th>isFraud</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>335623</td>
      <td>172717</td>
    </tr>
    <tr>
      <th>1</th>
      <td>669</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Well, not as informative as I'd hoped, but at least it's good to know that no fraudulent transactions went to merchants!</p>

</div>
</div>
</div>
    </div>
  </div>
</body>
</html>
