/*!
 * Valine v1.3.9
 * (c) 2017-2019 xCss
 * Released under the GPL-2.0 License.
 * Last Update: 2019-6-26 21:52:17
 */
!
function(e, t) {
  "object" == typeof exports && "object" == typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define([], t) : "object" == typeof exports ? exports.Valine = t() : e.Valine = t()
} (this,
function() {
  return function(e) {
    function t(r) {
      if (n[r]) return n[r].exports;
      var i = n[r] = {
        i: r,
        l: !1,
        exports: {}
      };
      return e[r].call(i.exports, i, i.exports, t),
      i.l = !0,
      i.exports
    }
    var n = {};
    return t.m = e,
    t.c = n,
    t.i = function(e) {
      return e
    },
    t.d = function(e, n, r) {
      t.o(e, n) || Object.defineProperty(e, n, {
        configurable: !1,
        enumerable: !0,
        get: r
      })
    },
    t.n = function(e) {
      var n = e && e.__esModule ? function() {
        return e.default
      }
      : function() {
        return e
      };
      return t.d(n, "a", n),
      n
    },
    t.o = function(e, t) {
      return Object.prototype.hasOwnProperty.call(e, t)
    },
    t.p = "",
    t(t.s = 17)
  } ([function(e, t, n) {
    var r, i, o;
    /*!
	autosize 4.0.2
	license: MIT
	http://www.jacklmoore.com/autosize
*/
    !
    function(n, a) {
      i = [e, t],
      r = a,
      void 0 !== (o = "function" == typeof r ? r.apply(t, i) : r) && (e.exports = o)
    } (0,
    function(e, t) {
      "use strict";
      function n(e) {
        function t(t) {
          var n = e.style.width;
          e.style.width = "0px",
          e.offsetWidth,
          e.style.width = n,
          e.style.overflowY = t
        }
        function n(e) {
          for (var t = []; e && e.parentNode && e.parentNode instanceof Element;) e.parentNode.scrollTop && t.push({
            node: e.parentNode,
            scrollTop: e.parentNode.scrollTop
          }),
          e = e.parentNode;
          return t
        }
        function r() {
          if (0 !== e.scrollHeight) {
            var t = n(e),
            r = document.documentElement && document.documentElement.scrollTop;
            e.style.height = "",
            e.style.height = e.scrollHeight + s + "px",
            l = e.clientWidth,
            t.forEach(function(e) {
              e.node.scrollTop = e.scrollTop
            }),
            r && (document.documentElement.scrollTop = r)
          }
        }
        function i() {
          r();
          var n = Math.round(parseFloat(e.style.height)),
          i = window.getComputedStyle(e, null),
          o = "content-box" === i.boxSizing ? Math.round(parseFloat(i.height)) : e.offsetHeight;
          if (o < n ? "hidden" === i.overflowY && (t("scroll"), r(), o = "content-box" === i.boxSizing ? Math.round(parseFloat(window.getComputedStyle(e, null).height)) : e.offsetHeight) : "hidden" !== i.overflowY && (t("hidden"), r(), o = "content-box" === i.boxSizing ? Math.round(parseFloat(window.getComputedStyle(e, null).height)) : e.offsetHeight), c !== o) {
            c = o;
            var s = a("autosize:resized");
            try {
              e.dispatchEvent(s)
            } catch(e) {}
          }
        }
        if (e && e.nodeName && "TEXTAREA" === e.nodeName && !o.has(e)) {
          var s = null,
          l = null,
          c = null,
          d = function() {
            e.clientWidth !== l && i()
          },
          u = function(t) {
            window.removeEventListener("resize", d, !1),
            e.removeEventListener("input", i, !1),
            e.removeEventListener("keyup", i, !1),
            e.removeEventListener("autosize:destroy", u, !1),
            e.removeEventListener("autosize:update", i, !1),
            Object.keys(t).forEach(function(n) {
              e.style[n] = t[n]
            }),
            o.delete(e)
          }.bind(e, {
            height: e.style.height,
            resize: e.style.resize,
            overflowY: e.style.overflowY,
            overflowX: e.style.overflowX,
            wordWrap: e.style.wordWrap
          });
          e.addEventListener("autosize:destroy", u, !1),
          "onpropertychange" in e && "oninput" in e && e.addEventListener("keyup", i, !1),
          window.addEventListener("resize", d, !1),
          e.addEventListener("input", i, !1),
          e.addEventListener("autosize:update", i, !1),
          e.style.overflowX = "hidden",
          e.style.wordWrap = "break-word",
          o.set(e, {
            destroy: u,
            update: i
          }),
          function() {
            var t = window.getComputedStyle(e, null);
            "vertical" === t.resize ? e.style.resize = "none": "both" === t.resize && (e.style.resize = "horizontal"),
            s = "content-box" === t.boxSizing ? -(parseFloat(t.paddingTop) + parseFloat(t.paddingBottom)) : parseFloat(t.borderTopWidth) + parseFloat(t.borderBottomWidth),
            isNaN(s) && (s = 0),
            i()
          } ()
        }
      }
      function r(e) {
        var t = o.get(e);
        t && t.destroy()
      }
      function i(e) {
        var t = o.get(e);
        t && t.update()
      }
      var o = "function" == typeof Map ? new